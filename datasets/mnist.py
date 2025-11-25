from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor


class FlowMatchingMNIST(Dataset):
    """Simple MNIST wrapper.

    Args:
        root: Where to save dataset.
        train: Whether it is train or valid dataset. Defaults to True.
        transform: Image transforms. Defaults to None.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        """Inits MNIST dataset wrapper."""
        self.dataset = MNIST(root=root, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        """Returns dataset length.

        Returns:
            Dataset length.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns MNIST dataset sample.

        Args:
            index: Sample index.

        Returns:
            Dict with sampled image.
        """
        image, _ = self.dataset[index]

        return {"image": image}


def get_mnist_dataloader(
    root: str = "/Share8/Michael_Bryant/MNIST/MNIST/",
    batch_size: int = 64,
) -> DataLoader:
    """Returns MNIST dataloader, prepared for generative model training.

    Args:
        root: Where to save dataset. Defaults to "./data".
        batch_size: Batch size. Defaults to 64.

    Returns:
        MNIST dataloader.
    """
    transforms = Compose(
        [
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),  # Normalize to [-1, 1]
        ]
    )

    dataset = FlowMatchingMNIST(root=root, train=True, transform=transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return dataloader
