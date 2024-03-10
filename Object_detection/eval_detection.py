import sys
import os
import argparse
import glob
import json

import torch
import torchvision

from datetime import datetime

from Data import dataloaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision

sys.path.append("..")
import utils

from timm import create_model


@torch.no_grad()
def test(model, device, test_loader, args):
    model.eval()
    map_ = MeanAveragePrecision()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = list(image.to(device) for image in data)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        output = model(data)
        map_.update(output, target)

    map_dict = map_.compute()
    if args.ss_framework:
        name = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}-dataset_{args.dataset}"
    else:
        name = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}-dataset_{args.dataset}"

    print_title = f"Object detection results for {name} @ {datetime.now()}"
    print_map = f"mAP@.5:.95: {map_dict['map'].item()}"
    print_map50 = f"mAP@.5: {map_dict['map_50'].item()}"
    print_map75 = f"mAP@.75: {map_dict['map_75'].item()}"
    print(print_title)
    print(print_map)
    print(print_map50)
    print(print_map75)
    with open("../eval_results.txt", "a") as f:
        f.write(print_title)
        f.write("\n")
        f.write(print_map)
        f.write("\n")
        f.write(print_map50)
        f.write("\n")
        f.write(print_map75)
        f.write("\n")


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


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        with open(args.root + "/bounding-boxes.json", "r") as f:
            targets = json.load(f)
        num_classes = 2  # polyp and background
        get_target_vals_fn = get_Kvasir_target_vals
    test_dataloader = dataloaders.get_test_dataloader(
        input_paths, targets, get_target_vals_fn, args.arch, 1024
    )

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        if args.arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                trainable_backbone_layers=5,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
        elif args.ss_framework == "mae":
            backbone = utils.get_MAE_backbone(
                None, False, None, False, None, True, 1024
            )
        elif args.ss_framework == "mocov3":
            backbone = utils.get_MoCoV3_backbone(
                None, args.arch, False, None, False, None, True, 1024
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
                False, None, False, None, True, ImageNet_weights=False, fixed_size=1024
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
        ckpt_path = f"Trained models/{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}-dataset_{args.dataset}.pth"
    else:
        ckpt_path = f"Trained models/{args.arch}-{args.pretraining}_init-frozen_{str(False)}-dataset_{args.dataset}.pth"

    main_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(main_dict["model_state_dict"])
    model.to(device)

    return (
        device,
        test_dataloader,
        model,
    )


def evaluate(args):

    (
        device,
        test_dataloader,
        model,
    ) = build(args)
    if not os.path.exists("../eval_results.txt"):
        open("../eval_results.txt", "w")

    test(model, device, test_dataloader, args)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned object detection model"
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
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()

    evaluate(args)


if __name__ == "__main__":
    main()

