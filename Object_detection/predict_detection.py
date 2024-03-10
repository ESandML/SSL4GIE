import sys
import os
import argparse
import numpy as np
import glob
import json
import cv2

import torch
import torchvision

from Data import dataloaders

sys.path.append("..")
import utils


@torch.no_grad()
def test(model, device, test_loader, args):
    model.eval()
    for i, (data, target, data0, p1, p2) in enumerate(test_loader):
        data = list(image.to(device) for image in data)
        output = model(data)
        output = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in output]

        target = target[0]
        output = output[0]
        data0 = data0[0]
        p1 = p1[0]
        p2 = p2[0]
        target["boxes"][:, 0] -= p1
        target["boxes"][:, 2] -= p1
        target["boxes"][:, 1] -= p2
        target["boxes"][:, 3] -= p2
        output["boxes"][:, 0] -= p1
        output["boxes"][:, 2] -= p1
        output["boxes"][:, 1] -= p2
        output["boxes"][:, 3] -= p2
        if args.arch == "vit_b":
            h0, w0 = data0.shape[-2:]
            if h0 > 1024 or w0 > 1024:
                target["boxes"] *= 2
                output["boxes"] *= 2

        numpy_image = data0.cpu().numpy() * 255
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        if args.idx:
            i = args.idx[i]
        if args.ss_framework:
            label = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}"
        else:
            label = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}"

        for box_idx in range(len(target["boxes"])):
            start = (
                int(target["boxes"][box_idx, 0].item()),
                int(target["boxes"][box_idx, 1].item()),
            )
            end = (
                int(target["boxes"][box_idx, 2].item()),
                int(target["boxes"][box_idx, 3].item()),
            )
            cv2_image = cv2.rectangle(cv2_image, start, end, (0, 234, 255), 6)
        for box_idx in range(len(output["boxes"])):
            start = (
                int(output["boxes"][box_idx, 0].item()),
                int(output["boxes"][box_idx, 1].item()),
            )
            end = (
                int(output["boxes"][box_idx, 2].item()),
                int(output["boxes"][box_idx, 3].item()),
            )
            cv2_image = cv2.rectangle(cv2_image, start, end, (0, 255, 0), 6)
        cv2.imwrite(f"Predictions {args.dataset}/test{i}_{label}.png", cv2_image)


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

    if args.idx:
        args.idx = args.idx.split(",")
        assert len(args.idx) > 0
    test_dataloader = dataloaders.get_test_dataloader(
        input_paths, targets, get_target_vals_fn, args.arch, 1024, args.idx, True, True
    )

    box_score_thresh = 0.5

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        if args.arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                trainable_backbone_layers=5,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                box_score_thresh=box_score_thresh,
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
                box_score_thresh=box_score_thresh,
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
                box_score_thresh=box_score_thresh,
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
            box_score_thresh=box_score_thresh,
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


def predict(args):

    if not os.path.exists(f"./Predictions {args.dataset}"):
        os.makedirs(f"./Predictions {args.dataset}")

    (
        device,
        test_dataloader,
        model,
    ) = build(args)

    test(model, device, test_dataloader, args)


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions with fine-tuned object detection model"
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
    parser.add_argument("--idx", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    predict(args)


if __name__ == "__main__":
    main()

