import random
from PIL import Image
import numpy as np

from torch.utils import data
import torchvision.transforms.functional as TF


def make_square(im, rgb=True):
    x, y = im.size
    size = max(x, y)
    mode = "RGB" if rgb else "I;16"
    fill_color = (0, 0, 0) if rgb else 0
    new_im = Image.new(mode, (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


class Dataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        eval_mode=False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.eval_mode = eval_mode
        if eval_mode:
            assert not (hflip or vflip)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x = make_square(Image.open(input_ID)).resize((224, 224))
        y_ = Image.open(target_ID)
        y = make_square(y_, rgb=False).resize((224, 224))
        y = np.array(y) / 65535

        x = self.transform_input(x)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.eval_mode:
            y_ = np.array(y_) / 65536
            y_ = self.transform_target(y_)
            return x.float(), y.float(), y_.float()
        else:
            return x.float(), y.float()

