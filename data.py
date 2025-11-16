# data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config as cfg

def get_transform():
    return transforms.Compose([transforms.ToTensor()])

def get_loaders():
    train_ds = datasets.MNIST(
        root=cfg.DATA_ROOT, train=True, download=True, transform=get_transform()
    )
    val_ds = datasets.MNIST(
        root=cfg.DATA_ROOT, train=False, download=True, transform=get_transform()
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader