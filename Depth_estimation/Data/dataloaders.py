import numpy as np
import random
import multiprocessing

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


def get_dataloaders(
    rank,
    world_size,
    train_rgb,
    train_depth,
    test_rgb,
    test_depth,
    val_rgb,
    val_depth,
    batch_size,
):

    transform_input4train = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_target = transforms.ToTensor()

    train_dataset = Dataset(
        input_paths=train_rgb,
        target_paths=train_depth,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
    )

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
            input_paths=test_rgb,
            target_paths=test_depth,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )

        val_dataset = Dataset(
            input_paths=val_rgb,
            target_paths=val_depth,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )

        test_dataloader = MultiEpochsDataLoader(
            dataset=test_dataset,
            batch_size=batch_size * 8,
            shuffle=False,
            num_workers=32,
        )

        val_dataloader = MultiEpochsDataLoader(
            dataset=val_dataset,
            batch_size=batch_size * 8,
            shuffle=False,
            num_workers=32,
        )
    else:
        test_dataloader = None
        val_dataloader = None

    return train_dataloader, test_dataloader, val_dataloader, train_sampler


def get_test_dataloader(test_rgb, test_depth):

    transform_target = transforms.ToTensor()

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = Dataset(
        input_paths=test_rgb,
        target_paths=test_depth,
        transform_input=transform_input4test,
        transform_target=transform_target,
        eval_mode=True,
    )

    test_dataloader = MultiEpochsDataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    return test_dataloader

