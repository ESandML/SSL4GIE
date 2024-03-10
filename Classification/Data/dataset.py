from PIL import Image

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        targets: list,
        transform_input=None,
    ):
        self.input_paths = input_paths
        self.targets = targets
        self.transform_input = transform_input

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target = self.targets[index]

        x = Image.open(input_ID).resize((224, 224))

        x = self.transform_input(x)
        return x.float(), target

