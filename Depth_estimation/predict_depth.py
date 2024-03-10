import sys
import os
import argparse
import glob
import cv2

import torch
import torchvision

from Data import dataloaders
from Metrics import losses

sys.path.append("..")
import utils


def cvt_map(depth, cmap="magma"):
    cv2_image = (255 - depth.numpy() * 255).astype("uint8")
    if cmap == "magma":
        cv2_cmap = cv2.COLORMAP_MAGMA
    elif cmap == "bone":
        cv2_cmap = cv2.COLORMAP_BONE
    return cv2.applyColorMap(cv2_image, cv2_cmap)


@torch.no_grad()
def test(model, device, test_loader, args):
    model.eval()
    for i, (data, target, target_og) in enumerate(test_loader):
        data = data.to(device)
        target = target.squeeze(1)
        output = model(data).squeeze(1).to("cpu")
        scale, shift = losses.compute_scale_and_shift(output, target, target > 0.0)
        output = scale.view(-1, 1, 1) * output + shift.view(-1, 1, 1)

        h, w = target_og.shape[2], target_og.shape[3]
        max_size = max(h, w)
        output = torchvision.transforms.functional.resize(
            output.unsqueeze(1), size=max_size
        )
        output = torchvision.transforms.functional.center_crop(output, (h, w)).squeeze()
        target_og = target_og.squeeze()

        output[output < 0.0] = 0.0
        output[output > 1.0] = 1.0
        output[target_og == 0.0] = 1.0
        target_og[target_og == 0.0] = 1.0
        diff = 3.5 * torch.abs(output - target_og)
        if diff.max() > 1:
            print("Maximum absolute error on scale exceeds 1")

        if args.idx:
            i = args.idx[i]
        if args.ss_framework:
            label = f"{args.arch}-{args.pretraining}_{args.ss_framework}_init-frozen_{str(False)}"
        else:
            label = f"{args.arch}-{args.pretraining}_init-frozen_{str(False)}"
        cv2.imwrite(f"Predictions {args.dataset}/test{i}_{label}.png", cvt_map(output))
        cv2.imwrite(f"Predictions {args.dataset}/GT{i}.png", cvt_map(target_og))
        cv2.imwrite(
            f"Predictions {args.dataset}/diff{i}_{label}.png",
            cvt_map(diff, cmap="bone"),
        )


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

    if args.idx:
        args.idx = args.idx.split(",")
        assert len(args.idx) > 0
        test_rgb = [test_rgb[int(i)] for i in args.idx]
        test_depth = [test_depth[int(i)] for i in args.idx]
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


def predict(args):

    if not os.path.exists(f"./Predictions {args.dataset}"):
        os.makedirs(f"./Predictions {args.dataset}")

    (test_dataloader, model, device) = build(args)

    test(model, device, test_dataloader, args)


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions with fine-tuned depth estimation model"
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
    parser.add_argument("--idx", type=str)

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

