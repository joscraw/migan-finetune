"""
Simple image dataset for MI-GAN fine-tuning.

Loads images from a directory, resizes to 512x512, normalizes to [-1, 1].
Generates random free-form masks on the fly using MI-GAN's mask generator.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Import MI-GAN's mask generation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.data_factory.ds_ffhq import RandomMask


class InpaintingDataset(Dataset):
    """
    Loads images from a directory and generates random masks.

    Returns:
        image: [3, 512, 512] float32 in [-1, 1]
        mask: [1, 512, 512] float32, 1=known, 0=hole
    """

    def __init__(self, image_dir, resolution=512, hole_range=(0.1, 0.6)):
        self.image_dir = image_dir
        self.resolution = resolution
        self.hole_range = list(hole_range)

        # Find all images
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in extensions:
                    self.image_paths.append(os.path.join(root, f))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # [0, 1]
        ])

        print(f"[Dataset] Loaded {len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        img = (img - 0.5) * 2  # [0,1] -> [-1,1]

        # Generate random mask (1=known, 0=hole)
        mask = RandomMask(self.resolution, hole_range=self.hole_range)
        mask = torch.from_numpy(mask)  # [1, H, W]

        return img, mask
