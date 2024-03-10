import sys
import os
import argparse
import glob
import random

from datetime import datetime

import torch
import torchvision

from Data import dataloaders
from Metrics import losses

sys.path.append("..")
import utils


def rmse(pred, targ):
    return torch.sqrt(torch.mean((pred - targ)[targ > 0] ** 2)).item()


def rel_err(pred, targ):
    return torch.median(torch.abs((pred - targ) / targ)[targ > 0]).item()


def abs_err(pred, targ):
    return torch.mean(torch.abs(pred - targ)[targ > 0]).item()


@torch.no_grad()
def test(model, device, test_loader, args):
    model.eval()
    rmse_accumulator = 0
    rel_err_accumulator = 0
    abs_err_accumulator = 0
    rmse_per_instance = []
    for i, (data, target, target_og) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device).squeeze(1)
        target_og = target_og.to(device)
        output = model(data).squeeze(1)
        scale, shift = losses.compute_scale_and_shift(output, target, target > 0.0)
        output = scale.view(-1, 1, 1) * output + shift.view(-1, 1, 1)
        h, w = target_og.shape[2], target_og.shape[3]
        max_size = max(h, w)
        output = torchvision.transforms.functional.resize(
            output.unsqueeze(1), size=max_size
        )
        output = torchvision.transforms.functional.center_crop(output, (h, w))
        output[output < 0.0] = 0.0
        output[output > 1.0] = 1.0
        output[target_og == 0.0] = 0.0

        if args.dataset == "C3VD":
            scale_ = 10
        output *= scale_
        target_og *= scale_
        rmse_accumulator += rmse(output, target_og)
        rel_err_accumulator += rel_err(output, target_og)
        abs_err_accumulator += abs_err(output, target_og)
        rmse_per_instance.append(str(rmse(output, target_og)))

    if args.ss_framework:
        name = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}-dataset_{args.dataset}"
    else:
        name = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}-dataset_{args.dataset}"
    print_title = f"Depth estimation results for {name} @ {datetime.now()}"
    print_rmse = f"RMSE: {rmse_accumulator/len(test_loader)}"
    print_rel_err = f"Relative Error: {rel_err_accumulator/len(test_loader)}"
    print_abs_err = f"Absolute Error: {abs_err_accumulator/len(test_loader)}"
    print(print_title)
    print(print_rmse)
    print(print_rel_err)
    print(print_abs_err)
    with open("../eval_results.txt", "a") as f:
        f.write(print_title)
        f.write("\n")
        f.write(print_rmse)
        f.write("\n")
        f.write(print_rel_err)
        f.write("\n")
        f.write(print_abs_err)
        f.write("\n")

    if args.print_list:
        if args.arch == "resnet50":
            arch_ID = "RN_"
        elif args.arch == "vit_b":
            arch_ID = "VT_"

        if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
            if args.pretraining == "Hyperkvasir":
                data_ID = "HK_"
            elif args.pretraining == "ImageNet_self":
                data_ID = "IN_"
            if args.ss_framework == "mocov3":
                alg_ID = "MC_"
            elif args.ss_framework == "barlowtwins":
                alg_ID = "BT_"
            elif args.ss_framework == "mae":
                alg_ID = "MA_"
            print(arch_ID + data_ID + alg_ID + args.dataset + " = [")
        elif args.pretraining == "ImageNet_class":
            print(arch_ID + "IN_SL_" + args.dataset + " = [")
        elif args.pretraining == "random":
            print(arch_ID + "NA_NA_" + args.dataset + " = [")
        for r in rmse_per_instance:
            print("    " + r + ",")
        print("]")


def list_rgb_and_depth_C3VD(vid_folder):
    rgb = []
    depth = []
    for vid in vid_folder:
        rgb_ = sorted(glob.glob(vid + "*color.png"))
        depth_ = sorted(glob.glob(vid + "*depth.tiff"))
        idx = argsort(
            [int(os.path.splitext(os.path.basename(f))[0][:-6]) for f in rgb_]
        )
        rgb += [rgb_[i] for i in idx]
        depth += [depth_[i] for i in idx]
    return rgb, depth


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "C3VD":
        test_vids = [
            args.root + "/trans_t2_b_under_review/t2v2/",
            args.root + "/cecum_t4_b_under_review/c4v3/",
        ]

        test_rgb, test_depth = list_rgb_and_depth_C3VD(test_vids)

    test_dataloader = dataloaders.get_test_dataloader(test_rgb, test_depth)

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        if args.ss_framework == "barlowtwins":
            model = utils.get_BarlowTwins_backbone(None, False, None, False, "depth")
        elif args.ss_framework == "mae":
            model = utils.get_MAE_backbone(None, False, None, False, "depth", False)
        elif args.ss_framework == "mocov3":
            model = utils.get_MoCoV3_backbone(
                None, args.arch, False, None, False, "depth", False
            )
    elif args.pretraining == "ImageNet_class":
        if args.arch == "resnet50":
            model = utils.get_ImageNet_or_random_ResNet(
                False, 1, False, "depth", ImageNet_weights=True
            )
        else:
            model = utils.get_ImageNet_or_random_ViT(
                False, 1, False, "depth", False, ImageNet_weights=True
            )
    elif args.pretraining == "random":
        if args.arch == "resnet50":
            model = utils.get_ImageNet_or_random_ResNet(
                False, 1, False, "depth", ImageNet_weights=False
            )
        else:
            model = utils.get_ImageNet_or_random_ViT(
                False, 1, False, "depth", False, ImageNet_weights=False
            )
    if args.ss_framework:
        ckpt_path = f"Trained models/{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}-dataset_{args.dataset}.pth"
    else:
        ckpt_path = f"Trained models/{args.arch}-{args.pretraining}_init-frozen_{str(False)}-dataset_{args.dataset}.pth"

    main_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(main_dict["model_state_dict"])
    model.to(device)

    return (test_dataloader, model, device)


def evaluate(args):

    (test_dataloader, model, device) = build(args)
    if not os.path.exists("../eval_results.txt"):
        open("../eval_results.txt", "w")

    test(model, device, test_dataloader, args)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned depth estimation model"
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
    parser.add_argument("--dataset", type=str, required=True, choices=["C3VD"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--print-list", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = get_args()
    evaluate(args)


if __name__ == "__main__":
    main()

