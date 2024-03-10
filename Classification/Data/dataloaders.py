import numpy as np

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from torch.utils.data.distributed import DistributedSampler

from Data.dataset import Dataset


class MultiEpochsDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(rank, world_size, input_paths, targets, batch_size):

    transform_input4train = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    train_dataset = Dataset(
        input_paths=input_paths,
        targets=targets,
        transform_input=transform_input4train,
    )

    train_dataset = data.Subset(train_dataset, train_indices)

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True
    )

    train_dataloader = MultiEpochsDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=8,
    )

    if rank == 0:

        transform_input4test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_dataset = Dataset(
            input_paths=input_paths,
            targets=targets,
            transform_input=transform_input4test,
        )

        val_dataset = Dataset(
            input_paths=input_paths,
            targets=targets,
            transform_input=transform_input4test,
        )
        val_dataset = data.Subset(val_dataset, val_indices)
        test_dataset = data.Subset(test_dataset, test_indices)

        test_dataloader = MultiEpochsDataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )

        val_dataloader = MultiEpochsDataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )
    else:
        test_dataloader = None
        val_dataloader = None

    return train_dataloader, test_dataloader, val_dataloader, train_sampler


def get_test_dataloader(input_paths, targets):

    _, test_indices, _ = split_ids(len(input_paths))

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = Dataset(
        input_paths=input_paths, targets=targets, transform_input=transform_input4test
    )

    test_dataset = data.Subset(test_dataset, test_indices)

    test_dataloader = MultiEpochsDataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    return test_dataloader

