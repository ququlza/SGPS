"""RGB image loading for SGPS, preserving source image IDs and transforms."""
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root='dataset/demo', resolution=256, device='cuda', start_id=None, end_id=None):
        # Define the file extensions to search for
        # A single scan avoids matching every file twice on Windows.
        extensions = {'.jpg', '.jpeg', '.png'}
        self.data = sorted(file for file in Path(root).rglob('*')
                           if file.is_file() and file.suffix.lower() in extensions)
        normalized = [str(file.resolve()).casefold() for file in self.data]
        if len(normalized) != len(set(normalized)):
            raise ValueError('Duplicate image paths')

        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def __getitem__(self, i):
        with Image.open(self.data[i]) as source:
            img = (self.trans(source.convert('RGB')) * 2 - 1).to(self.device)
        return img

    def __len__(self):
        return len(self.data)

    def get_data(self, size=16, sigma=0):
        data = torch.stack([self.__getitem__(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma
