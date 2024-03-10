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
    mf1 = performance.meanF1Score(n_class=args.n_class)
    mprec = performance.meanPrecision(n_class=args.n_class)
    mrec = performance.meanRecall(n_class=args.n_class)
    for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        if i == 0:
            pred = torch.argmax(output, 1)
            targ = target
        else:
            pred = torch.cat((pred, torch.argmax(output, 1)), 0)
            targ = torch.cat((targ, target), 0)

    if args.ss_framework:
        name = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}-dataset_{args.dataset}"
    else:
        name = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}-dataset_{args.dataset}"
    print_title = f"Classification results for {name} @ {datetime.now()}"
    print_mf1 = f"mF1: {mf1(pred, targ).item()}"
    print_mprec = f"mPrecision: {mprec(pred, targ).item()}"
    print_mrec = f"mRecall: {mrec(pred, targ).item()}"
    print_acc = f"Accuracy: {(pred==targ).sum().item()/len(pred)}"
    print(print_title)
    print(print_mf1)
    print(print_mprec)
    print(print_mrec)
    print(print_acc)
    with open("../eval_results.txt", "a") as f:
        f.write(print_title)
        f.write("\n")
        f.write(print_mf1)
        f.write("\n")
        f.write(print_mprec)
        f.write("\n")
        f.write(print_mrec)
        f.write("\n")
        f.write(print_acc)
        f.write("\n")


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset.startswith("Hyperkvasir"):
        if args.dataset.endswith("pathological"):
            n_class = 12
            class_type = "pathological-findings/"
        elif args.dataset.endswith("anatomical"):
            n_class = 6
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
        for ssf in subsub_folders:
            contents = sorted(glob.glob(ssf + "*.jpg"))
            ssf_targets = [class_id for _ in range(len(contents))]
            input_paths += contents
            targets += ssf_targets
            class_id += 1
        n_class = class_id
        args.n_class = n_class

    test_dataloader = dataloaders.get_test_dataloader(input_paths, targets)

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        if args.ss_framework == "barlowtwins":
            model = utils.get_BarlowTwins_backbone(None, True, n_class, False, None)
        elif args.ss_framework == "mae":
            model = utils.get_MAE_backbone(None, True, n_class, False, None, False)
        elif args.ss_framework == "mocov3":
            model = utils.get_MoCoV3_backbone(
                None, args.arch, True, n_class, False, None, False
            )
    elif args.pretraining == "ImageNet_class":
        if args.arch == "resnet50":
            model = utils.get_ImageNet_or_random_ResNet(
                True, n_class, False, None, ImageNet_weights=True
            )
        else:
            model = utils.get_ImageNet_or_random_ViT(
                True, n_class, False, None, False, ImageNet_weights=True
            )
    elif args.pretraining == "random":
        if args.arch == "resnet50":
            model = utils.get_ImageNet_or_random_ResNet(
                True, n_class, False, None, ImageNet_weights=False
            )
        else:
            model = utils.get_ImageNet_or_random_ViT(
                True, n_class, False, None, False, ImageNet_weights=False
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
        description="Evaluate fine-tuned classification model"
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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Hyperkvasir_pathological", "Hyperkvasir_anatomical"],
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    evaluate(args)


if __name__ == "__main__":
    main()

