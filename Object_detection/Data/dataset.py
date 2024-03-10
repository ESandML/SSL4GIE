import random
from PIL import Image

import torch
from torch.utils import data
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


class Dataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        targets: dict,
        get_target_vals_fn,
        transform_input=None,
        hflip=False,
        vflip=False,
        rotate=False,
        arch="resnet50",
        fixed_size=None,
        post_process=False,
    ):
        self.input_paths = input_paths
        self.targets = targets
        self.get_target_vals_fn = get_target_vals_fn
        self.transform_input = transform_input
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.arch = arch
        self.fixed_size = fixed_size
        self.post_process = post_process

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target = self.get_target_vals_fn(input_ID, self.targets)

        x = Image.open(input_ID)

        x = self.transform_input(x)

        H, W = x.shape[1], x.shape[2]
        if self.post_process:
            x0 = x

        if self.rotate:
            if random.uniform(0.0, 1.0) > 0.5:
                x = torch.rot90(x, dims=[1, 2])
                for i in range(len(target["labels"])):
                    new_xmin = target["boxes"][i, 1]
                    new_xmax = target["boxes"][i, 3]
                    new_ymin = W - target["boxes"][i, 2]
                    new_ymax = W - target["boxes"][i, 0]
                    target["boxes"][i, 0] = new_xmin
                    target["boxes"][i, 2] = new_xmax
                    target["boxes"][i, 1] = new_ymin
                    target["boxes"][i, 3] = new_ymax
                H, W = x.shape[1], x.shape[2]

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                for i in range(len(target["labels"])):
                    new_xmin = W - target["boxes"][i, 2]
                    new_xmax = W - target["boxes"][i, 0]
                    target["boxes"][i, 0] = new_xmin
                    target["boxes"][i, 2] = new_xmax

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                for i in range(len(target["labels"])):
                    new_ymin = H - target["boxes"][i, 3]
                    new_ymax = H - target["boxes"][i, 1]
                    target["boxes"][i, 1] = new_ymin
                    target["boxes"][i, 3] = new_ymax

        if self.arch != "resnet50":
            if H > self.fixed_size or W > self.fixed_size:
                if H % 2 != 0:
                    x = TF.pad(x, (0, 0, 0, 1))
                    H += 1
                if W % 2 != 0:
                    x = TF.pad(x, (0, 0, 1, 0))
                    W += 1
                x = TF.resize(
                    x,
                    size=(H // 2, W // 2),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                H, W = x.shape[1], x.shape[2]
                target["boxes"] /= 2
            p1 = torch.floor(torch.tensor((self.fixed_size - W) / 2)).int().item()
            p2 = torch.floor(torch.tensor((self.fixed_size - H) / 2)).int().item()
            p3 = torch.ceil(torch.tensor((self.fixed_size - W) / 2)).int().item()
            p4 = torch.ceil(torch.tensor((self.fixed_size - H) / 2)).int().item()
            x = TF.pad(x, (p1, p2, p3, p4))
            target["boxes"][:, 0] += p1
            target["boxes"][:, 2] += p1
            target["boxes"][:, 1] += p2
            target["boxes"][:, 3] += p2
        elif self.post_process:
            p1 = 0
            p2 = 0
        if self.post_process:
            return x.float(), target, x0.float(), p1, p2
        else:
            return x.float(), target

