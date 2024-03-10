import sys
import os
import argparse
import glob
from skimage.io import imsave

import torch
import torchvision
import segmentation_models_pytorch as smp

from Data import dataloaders

sys.path.append("..")
import utils


@torch.no_grad()
def test(model, device, test_loader, args):
    model.eval()
    for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        output = model(data).to("cpu")
        h, w = target.shape[2], target.shape[3]
        output = (
            torchvision.transforms.functional.resize(output, size=(h, w)).sigmoid()
            > 0.5
        )
        if args.idx:
            i = args.idx[i]
        if args.ss_framework:
            label = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}"
        else:
            label = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}"

        imsave(
            f"Predictions {args.dataset}/test{i}_{label}.png",
            (output.squeeze().numpy() * 255).astype("uint8"),
        )


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

    if args.idx:
        args.idx = args.idx.split(",")
        assert len(args.idx) > 0
    test_dataloader = dataloaders.get_test_dataloader(
        input_paths, target_paths, args.idx, True
    )

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


def predict(args):

    if not os.path.exists(f"./Predictions {args.dataset}"):
        os.makedirs(f"./Predictions {args.dataset}")

    (test_dataloader, model, device) = build(args)

    test(model, device, test_dataloader, args)


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions with fine-tuned segmentation model"
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
    parser.add_argument("--idx", type=str)

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

