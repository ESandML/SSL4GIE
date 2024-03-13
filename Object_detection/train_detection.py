import sys
import os
import argparse
import time
import numpy as np
import glob
import random
import json

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from Data import dataloaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision

sys.path.append("..")
import utils


def reduce_dict(input_dict, world_size, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train_epoch(
    model,
    rank,
    world_size,
    train_loader,
    train_sampler,
    optimizer,
    epoch,
    log_path,
    accum_iter,
    scaler,
):
    t = time.time()
    model.train()
    loss_accumulator = []
    optimizer.zero_grad()
    train_sampler.set_epoch(epoch - 1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = list(image.cuda(rank) for image in data)
        target = [{k: v.cuda(rank) for k, v in t.items()} for t in target]

        with torch.cuda.amp.autocast():
            loss_dict = model(data, target)
            loss = sum(loss for loss in loss_dict.values()) / accum_iter
        loss_dict_reduced = reduce_dict(loss_dict, world_size)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values()) / accum_iter
        loss_accumulator.append(losses_reduced.item())
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0:
                if (batch_idx + 1) // accum_iter < len(train_loader) // accum_iter:
                    print(
                        "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                            epoch,
                            (batch_idx + 1) * len(data) * world_size,
                            len(train_loader.dataset),
                            100.0 * (batch_idx + 1) / len(train_loader),
                            np.mean(loss_accumulator[-accum_iter:]),
                            time.time() - t,
                        ),
                        end="",
                    )
                else:
                    printout = "Train Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(data) * world_size,
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        np.mean(loss_accumulator),
                        time.time() - t,
                    )
                    print("\r" + printout)
                    with open(log_path, "a") as f:
                        f.write(printout)
                        f.write("\n")
        dist.barrier()
    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, rank, test_loader, epoch, log_path, metric):
    t = time.time()
    model.eval()
    N = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = list(image.cuda(rank) for image in data)
        target = [{k: v.cuda(rank) for k, v in t.items()} for t in target]
        N += len(data)
        output = model(data)
        metric.update(output, target)
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tTime: {:.6f}".format(
                    epoch,
                    N,
                    len(test_loader.dataset),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    time.time() - t,
                ),
                end="",
            )
        else:
            metric_dict = metric.compute()
            printout = "Test  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                epoch,
                N,
                len(test_loader.dataset),
                100.0 * (batch_idx + 1) / len(test_loader),
                metric_dict["map"].item(),
                time.time() - t,
            )

            print("\r" + printout)
            with open(log_path, "a") as f:
                f.write(printout)
                f.write("\n")
    metric.reset()
    return metric_dict["map"]


def get_Kvasir_target_vals(input_path, targets):
    objects = targets[os.path.splitext(os.path.basename(input_path))[0]]["bbox"]
    bboxes = torch.zeros((len(objects), 4), dtype=torch.float32)
    labels = torch.ones((len(objects),), dtype=torch.int64)
    for i, obj in enumerate(objects):
        bboxes[i, 0] = objects[i]["xmin"]
        bboxes[i, 1] = objects[i]["ymin"]
        bboxes[i, 2] = objects[i]["xmax"]
        bboxes[i, 3] = objects[i]["ymax"]
    target_vals = {}
    target_vals["boxes"] = bboxes
    target_vals["labels"] = labels
    return target_vals


def build(args, rank):

    if args.dataset == "Kvasir":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        with open(args.root + "/bounding-boxes.json", "r") as f:
            targets = json.load(f)
        num_classes = 2
        get_target_vals_fn = get_Kvasir_target_vals
    (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
    ) = dataloaders.get_dataloaders(
        rank,
        args.world_size,
        input_paths,
        targets,
        get_target_vals_fn,
        args.batch_size // args.world_size // args.accum_iter,
        args.arch,
        1024,
    )

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        assert os.path.exists(args.ckpt)
        if args.arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                trainable_backbone_layers=5,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
            weights = torch.load(args.ckpt, map_location="cpu")
            model.backbone.body.load_state_dict(weights)
        elif args.ss_framework == "mae":
            backbone = utils.get_MAE_backbone(
                args.ckpt, False, None, args.frozen, None, True, 1024
            )
        elif args.ss_framework == "mocov3":
            backbone = utils.get_MoCoV3_backbone(
                args.ckpt, args.arch, False, None, args.frozen, None, True, 1024
            )
    elif args.pretraining == "ImageNet_class":
        if args.arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                trainable_backbone_layers=5,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            backbone = utils.get_ImageNet_or_random_ViT(
                False, None, False, None, True, ImageNet_weights=True, fixed_size=1024
            )
    elif args.pretraining == "random":
        if args.arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                pretrained_backbone=False,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        else:
            backbone = utils.get_ImageNet_or_random_ViT(
                False,
                None,
                args.frozen,
                None,
                True,
                ImageNet_weights=False,
                fixed_size=1024,
            )
    if args.arch == "vit_b":
        model = torchvision.models.detection.faster_rcnn.FasterRCNN(
            backbone,
            num_classes=num_classes,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        model.transform.fixed_size = (1024, 1024)
    if args.ss_framework:
        ckpt_path = f"Trained models/{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(args.frozen)}-dataset_{args.dataset}.pth"
        log_path = f"Trained models/{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(args.frozen)}-dataset_{args.dataset}.txt"
    else:
        ckpt_path = f"Trained models/{args.arch}-{args.pretraining}_init-frozen_{str(args.frozen)}-dataset_{args.dataset}.pth"
        log_path = f"Trained models/{args.arch}-{args.pretraining}_init-frozen_{str(args.frozen)}-dataset_{args.dataset}.txt"

    if os.path.exists(ckpt_path):
        main_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(main_dict["model_state_dict"])
        start_epoch = main_dict["epoch"] + 1
        prev_best_test = main_dict["val_perf"]
        random.setstate(main_dict["py_state"])
        np.random.set_state(main_dict["np_state"])
        torch.set_rng_state(main_dict["torch_state"])
    else:
        start_epoch = 1
        prev_best_test = None
        open(log_path, "w")

    model.cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.arch == "resnet50":

        model = DDP(model, device_ids=[rank])
    elif args.arch == "vit_b":
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    if prev_best_test is not None:
        optimizer.load_state_dict(main_dict["optimizer_state_dict"])
        scaler.load_state_dict(main_dict["scaler_state_dict"])

    return (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
        model,
        optimizer,
        ckpt_path,
        log_path,
        start_epoch,
        prev_best_test,
        scaler,
    )


def train(rank, args):

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=args.world_size,
        init_method="tcp://localhost:58475",
    )

    print(f"Rank {rank + 1}/{args.world_size} process initialized.\n")

    if rank == 0:
        if not os.path.exists("./Trained models"):
            os.makedirs("./Trained models")

    (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
        model,
        optimizer,
        ckpt_path,
        log_path,
        start_epoch,
        prev_best_test,
        scaler,
    ) = build(args, rank)

    if rank == 0:
        metric = MeanAveragePrecision(sync_on_compute=False)
        with open(log_path, "a") as f:
            f.write(str(args))
            f.write("\n")
    dist.barrier()

    if args.lrs:
        val_perf = torch.tensor(0).cuda(rank)
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5
            )
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        if prev_best_test is not None:
            sched_dict = scheduler.state_dict()
            sched_dict["best"] = prev_best_test
            sched_dict["last_epoch"] = start_epoch - 1
            sched_dict["_last_lr"] = [lr]
            scheduler.load_state_dict(sched_dict)

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            loss = train_epoch(
                model,
                rank,
                args.world_size,
                train_dataloader,
                train_sampler,
                optimizer,
                epoch,
                log_path,
                args.accum_iter,
                scaler,
            )
            if rank == 0:
                val_perf = test(
                    model.module, rank, val_dataloader, epoch, log_path, metric
                ).cuda(rank)
                test_perf = test(
                    model.module, rank, test_dataloader, epoch, log_path, metric
                )
            dist.barrier()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs:
            torch.distributed.broadcast(val_perf, 0)
            scheduler.step(val_perf)
            if rank == 0:
                if lr != optimizer.state_dict()["param_groups"][0]["lr"]:
                    lr = optimizer.state_dict()["param_groups"][0]["lr"]
                    with open(log_path, "a") as f:
                        printout = "Epoch    {}: reducing learning rate of group 0 to {}.".format(
                            epoch, lr
                        )
                        print(printout)
                        f.write(printout)
                        f.write("\n")
        if rank == 0:
            if prev_best_test == None or val_perf > prev_best_test:
                print("Saving...")
                with open(log_path, "a") as f:
                    f.write("Saving...")
                    f.write("\n")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "loss": loss,
                        "val_perf": val_perf.item(),
                        "test_perf": test_perf.item(),
                        "py_state": random.getstate(),
                        "np_state": np.random.get_state(),
                        "torch_state": torch.get_rng_state(),
                    },
                    ckpt_path,
                )
                prev_best_test = val_perf.item()
        dist.barrier()
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained model for object detection"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["resnet50", "vit_s", "vit_b", "vit_l"],
        dest="arch",
    )
    parser.add_argument(
        "--pretraining",
        type=str,
        required=True,
        choices=["Hyperkvasir", "ImageNet_class", "ImageNet_self", "random"],
    )
    parser.add_argument(
        "--ss-framework", type=str, choices=["barlowtwins", "mocov3", "mae"]
    )
    parser.add_argument("--checkpoint", type=str, dest="ckpt")
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", action="store_true", default=False, dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )

    return parser.parse_args()


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    args = get_args()
    args.world_size = torch.cuda.device_count()
    assert args.batch_size % args.world_size == 0
    mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)


if __name__ == "__main__":
    main()

