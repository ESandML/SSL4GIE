import sys
import os
import argparse
import time
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from Data import dataloaders
from Metrics import performance

sys.path.append("..")
import utils


def train_epoch(
    model,
    rank,
    world_size,
    train_loader,
    train_sampler,
    optimizer,
    epoch,
    loss_fn,
    log_path,
    scaler,
):
    t = time.time()
    model.train()
    train_sampler.set_epoch(epoch - 1)
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(rank), target.cuda(rank)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        dist.all_reduce(loss)
        loss /= world_size
        loss_accumulator.append(loss.item())
        if rank == 0:
            if batch_idx + 1 < len(train_loader):
                print(
                    "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(data) * world_size,
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        loss.item(),
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
def test(model, rank, test_loader, epoch, perf_fn, log_path):
    t = time.time()
    model.eval()
    perf_accumulator = 0
    N = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(rank), target.cuda(rank)
        N += len(data)
        output = model(data)
        if batch_idx == 0:
            pred = torch.argmax(output, 1)
            targ = target
        else:
            pred = torch.cat((pred, torch.argmax(output, 1)), 0)
            targ = torch.cat((targ, target), 0)
        perf = perf_fn(pred, targ).item()
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    N,
                    len(test_loader.dataset),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    perf,
                    time.time() - t,
                ),
                end="",
            )
        else:
            printout = "Test  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                epoch,
                N,
                len(test_loader.dataset),
                100.0 * (batch_idx + 1) / len(test_loader),
                perf,
                time.time() - t,
            )

            print("\r" + printout)
            with open(log_path, "a") as f:
                f.write(printout)
                f.write("\n")
    return perf


def build(args, rank):

    if args.dataset.startswith("Hyperkvasir"):
        if args.dataset.endswith("pathological"):
            class_type = "pathological-findings/"
        elif args.dataset.endswith("anatomical"):
            class_type = "anatomical-landmarks/"
        base_folders = sorted(glob.glob(args.root + "/labeled-images/*/"))
        sub_folders = []
        for bf in base_folders:
            sub_folders += sorted(glob.glob(bf + "*/"))
        subsub_folders = []
        for sf in sub_folders:
            if sf.endswith(class_type):
                subsub_folders += sorted(glob.glob(sf + "*/"))
        class_id = 0
        input_paths = []
        targets = []
        N_in_class = []
        N_total = 0
        for ssf in subsub_folders:
            contents = sorted(glob.glob(ssf + "*.jpg"))
            ssf_targets = [class_id for _ in range(len(contents))]
            input_paths += contents
            targets += ssf_targets
            class_id += 1
            N_in_class.append(len(contents))
            N_total += len(contents)
        n_class = class_id
        class_weights = [1 / N * N_total / n_class for N in N_in_class]
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
        batch_size=args.batch_size // args.world_size,
    )

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        assert os.path.exists(args.ckpt)
        if args.ss_framework == "barlowtwins":
            model = utils.get_BarlowTwins_backbone(
                args.ckpt, True, n_class, args.frozen, None
            )
        elif args.ss_framework == "mae":
            model = utils.get_MAE_backbone(
                args.ckpt, True, n_class, args.frozen, None, False
            )
        elif args.ss_framework == "mocov3":
            model = utils.get_MoCoV3_backbone(
                args.ckpt, args.arch, True, n_class, args.frozen, None, False
            )
    elif args.pretraining == "ImageNet_class":
        if args.arch == "resnet50":
            model = utils.get_ImageNet_or_random_ResNet(
                True, n_class, args.frozen, None, ImageNet_weights=True
            )
        else:
            model = utils.get_ImageNet_or_random_ViT(
                True, n_class, args.frozen, None, False, ImageNet_weights=True
            )
    elif args.pretraining == "random":
        if args.arch == "resnet50":
            model = utils.get_ImageNet_or_random_ResNet(
                True, n_class, args.frozen, None, ImageNet_weights=False
            )
        else:
            model = utils.get_ImageNet_or_random_ViT(
                True, n_class, args.frozen, None, False, ImageNet_weights=False
            )
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
        class_weights,
        scaler,
    )


def train(rank, args):

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=args.world_size,
        init_method="tcp://localhost:58472",
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
        class_weights,
        scaler,
    ) = build(args, rank)

    loss_fn = nn.CrossEntropyLoss(torch.tensor(class_weights).cuda(rank))
    perf_fn = performance.meanF1Score(n_class=len(class_weights))

    if rank == 0:
        with open(log_path, "a") as f:
            f.write(str(args))
            f.write("\n")
    dist.barrier()

    if args.lrs:
        val_perf_ = torch.tensor(0).cuda(rank)
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
                loss_fn,
                log_path,
                scaler,
            )
            if rank == 0:
                val_perf = test(
                    model.module, rank, val_dataloader, epoch, perf_fn, log_path
                )
                if args.lrs:
                    val_perf_ = torch.tensor(val_perf).cuda(rank)
                test_perf = test(
                    model.module, rank, test_dataloader, epoch, perf_fn, log_path
                )
            dist.barrier()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs:
            torch.distributed.broadcast(val_perf_, 0)
            scheduler.step(val_perf_)
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
                        "val_perf": val_perf,
                        "test_perf": test_perf,
                        "py_state": random.getstate(),
                        "np_state": np.random.get_state(),
                        "torch_state": torch.get_rng_state(),
                    },
                    ckpt_path,
                )
                prev_best_test = val_perf
        dist.barrier()
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained model for classification"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["resnet50", "vit_b"],
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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Hyperkvasir_pathological", "Hyperkvasir_anatomical"],
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
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

