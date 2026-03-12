import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

CIFAR10_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog",
]


def get_cifar10_loaders(data_root, batch_size, num_workers, distributed=False):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_cifar100_loaders(data_root, batch_size, num_workers, distributed=False):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = torchvision.datasets.CIFAR100(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


class CIFAR10CDataset(Dataset):
    """Single corruption type from CIFAR-10-C."""
    def __init__(self, root, corruption, severity=1, transform=None):
        assert 1 <= severity <= 5
        data_path   = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")
        data   = np.load(data_path)
        labels = np.load(labels_path)
        idx = slice((severity - 1) * 10000, severity * 10000)
        self.data   = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_cifar10c_loader(root, corruption, severity, batch_size, num_workers):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = CIFAR10CDataset(root, corruption, severity, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)