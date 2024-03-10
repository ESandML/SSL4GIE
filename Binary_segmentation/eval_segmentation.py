import sys
import os
import argparse
import glob

from datetime import datetime

import torch
import torchvision

import segmentation_models_pytorch as smp

from Data import dataloaders
from Metrics import performance

sys.path.append("..")
import utils


@torch.no_grad()
def test(model, device, test_loader, args):
    model.eval()
    dice = performance.DiceScore()
    iou = performance.IoU()
    prec = performance.Precision()
    rec = performance.Recall()
    dice_accumulator = 0
    iou_accumulator = 0
    prec_accumulator = 0
    rec_accumulator = 0
    dice_per_instance = []
    for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        h, w = target.shape[2], target.shape[3]
        output = torchvision.transforms.functional.resize(output, size=(h, w))

        dice_accumulator += dice(output, target).item()
        iou_accumulator += iou(output, target).item()
        prec_accumulator += prec(output, target).item()
        rec_accumulator += rec(output, target).item()
        dice_per_instance.append(str(dice(output, target).item()))

    if args.ss_framework:
        name = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}-dataset_{args.dataset}"
    else:
        name = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}-dataset_{args.dataset}"
    print_title = f"Segmentation results for {name} @ {datetime.now()}"
    print_dice = f"Dice: {dice_accumulator/len(test_loader)}"
    print_iou = f"IoU: {iou_accumulator/len(test_loader)}"
    print_prec = f"Precision: {prec_accumulator/len(test_loader)}"
    print_rec = f"Recall: {rec_accumulator/len(test_loader)}"
    print(print_title)
    print(print_dice)
    print(print_iou)
    print(print_prec)
    print(print_rec)
    with open("../eval_results.txt", "a") as f:
        f.write(print_title)
        f.write("\n")
        f.write(print_dice)
        f.write("\n")
        f.write(print_iou)
        f.write("\n")
        f.write(print_prec)
        f.write("\n")
        f.write(print_rec)
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
        for r in dice_per_instance:
            print("    " + r + ",")
        print("]")


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "/Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))

    test_dataloader = dataloaders.get_test_dataloader(input_paths, target_paths)

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        if args.arch == "resnet50":
            model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None)
        elif args.ss_framework == "mae":
            model = utils.get_MAE_backbone(None, False, 1, False, "seg", False)
        elif args.ss_framework == "mocov3":
            model = utils.get_MoCoV3_backbone(
                None, args.arch, False, 1, False, "seg", False
            )
    elif args.pretraining == "ImageNet_class":
        if args.arch == "resnet50":
            model = smp.DeepLabV3Plus(encoder_name="resnet50")
        else:
            model = utils.get_ImageNet_or_random_ViT(
                False, 1, False, "seg", False, ImageNet_weights=True
            )
    elif args.pretraining == "random":
        if args.arch == "resnet50":
            model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None)
        else:
            model = utils.get_ImageNet_or_random_ViT(
                False, 1, False, "seg", False, ImageNet_weights=False
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
        description="Evaluate fine-tuned segmentation model"
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
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--print-list", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = get_args()
    evaluate(args)


if __name__ == "__main__":
    main()

