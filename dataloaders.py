# Harbin Institute of Technology Bachelor Thesis
# Author: HIT Michael_Bryant
# Mail: 1137892110@qq.com

import os
from tempfile import TemporaryFile
from unittest import TestLoader
from torch.utils.data.dataset import Subset
from types import SimpleNamespace
import numpy as np


from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Gone
# Import own files
from metadata_manager import *

# TODO: commented out because cv2 problems
from datasets.dataset_LIDC import *
from datasets.dataset_isic import *
from datasets.dataset_isic_style import *


def get_dataloader(
    task, split, batch_size, shuffle=True, splitratio=[0.6, 0.2, 0.2], randomsplit=False
):
    """
    Returns dataloader for training/validation/testing

    Args
        task: (string) which dataset/task for constructing dataloader
        split: (string) train/val/test
        batch_size: (int) batch size
        shuffle: (Bool) data randomly ordered?
    """
    meta_dict = get_meta(task)
    meta = SimpleNamespace(**meta_dict)

    if task == "LIDC":
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = LIDC(trafo, apply_symmetric_transforms=False, data_path=data_path)
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                #LIDC(trafo, apply_symmetric_transforms=True, data_path=data_path),
                dataset,
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                #LIDC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                dataset,
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                #LIDC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                dataset,
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )
    if task == "isic3_style_concat":
        # this concatenates the train splits from isic3_style0, isic3_style1, isic3_style2 so that the style probabilistic U-Net does not see the test data during training

        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])

        # Build dataset style 0 only
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=0
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style0_train = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=0
                ),
                train_indices,
            )

        if split == "val":
            dataset_style0_val = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style0_test = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                test_indices,
            )

        # Build dataset style 1 only
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=1
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style1_train = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=1
                ),
                train_indices,
            )

        if split == "val":
            dataset_style1_val = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style1_test = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                test_indices,
            )

        # Build dataset style 2 only
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=2
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style2_train = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=2
                ),
                train_indices,
            )

        if split == "val":
            dataset_style2_val = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style2_test = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                test_indices,
            )

        # Concatenate the datasets
        if split == "train":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_train, dataset_style1_train, dataset_style2_train]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_val, dataset_style1_val, dataset_style2_val]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_test, dataset_style1_test, dataset_style2_test]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )
    return dataloader, dataset


def get_dataloader_2(
    task, split, batch_size, shuffle=True, splitratio=[0.8, 0.0, 0.2], randomsplit=False
):
    """
    Returns dataloader for training/validation/testing

    Args
        task: (string) which dataset/task for constructing dataloader
        split: (string) train/val/test
        batch_size: (int) batch size
        shuffle: (Bool) data randomly ordered?
    """
    meta_dict = get_meta(task)
    meta = SimpleNamespace(**meta_dict)

    if task == "LIDC":
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = LIDC2(trafo, apply_symmetric_transforms=False, data_path=data_path)
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                #LIDC(trafo, apply_symmetric_transforms=True, data_path=data_path),
                dataset,
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                #LIDC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                dataset,
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                #LIDC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                dataset,
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )
    if task == "isic3_style_concat":
        # this concatenates the train splits from isic3_style0, isic3_style1, isic3_style2 so that the style probabilistic U-Net does not see the test data during training

        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])

        # Build dataset style 0 only
        dataset = ISIC_style_subset2(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=0
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style0_train = Subset(
                ISIC_style_subset2(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=0
                ),
                train_indices,
            )

        if split == "val":
            dataset_style0_val = Subset(
                ISIC_style_subset2(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style0_test = Subset(
                ISIC_style_subset2(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                test_indices,
            )

        # Build dataset style 1 only
        dataset = ISIC_style_subset2(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=1
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style1_train = Subset(
                ISIC_style_subset2(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=1
                ),
                train_indices,
            )

        if split == "val":
            dataset_style1_val = Subset(
                ISIC_style_subset2(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style1_test = Subset(
                ISIC_style_subset2(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                test_indices,
            )

        # Build dataset style 2 only
        dataset = ISIC_style_subset2(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=2
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style2_train = Subset(
                ISIC_style_subset2(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=2
                ),
                train_indices,
            )

        if split == "val":
            dataset_style2_val = Subset(
                ISIC_style_subset2(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style2_test = Subset(
                ISIC_style_subset2(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                test_indices,
            )

        # Concatenate the datasets
        if split == "train":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_train, dataset_style1_train, dataset_style2_train]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_val, dataset_style1_val, dataset_style2_val]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_test, dataset_style1_test, dataset_style2_test]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )
    return dataloader, dataset
