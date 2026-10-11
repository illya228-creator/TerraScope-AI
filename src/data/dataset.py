from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRDataset(Dataset):
    """Dataset for paired satellite images and binary change masks."""

    def __init__(self, root, split):
        self.root = Path(root) / split

        self.a_dir = self.root / "A"
        self.b_dir = self.root / "B"
        self.label_dir = self.root / "label"

        names_a = {
            p.name for p in self.a_dir.iterdir() if p.is_file()
        }
        names_b = {
            p.name for p in self.b_dir.iterdir() if p.is_file()
        }
        names_label = {
            p.name for p in self.label_dir.iterdir() if p.is_file()
        }

        if names_a != names_b or names_a != names_label:
            raise ValueError(
                f"{split}: filenames in A, B and label do not match"
            )

        self.names = sorted(names_a)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        with Image.open(self.a_dir / name) as image:
            image_a = np.array(image.convert("RGB"))

        with Image.open(self.b_dir / name) as image:
            image_b = np.array(image.convert("RGB"))

        with Image.open(self.label_dir / name) as image:
            mask = np.array(image.convert("L"))

        image_a = (
            torch.from_numpy(image_a.copy())
            .permute(2, 0, 1)
            .float() / 255.0
        )

        image_b = (
            torch.from_numpy(image_b.copy())
            .permute(2, 0, 1)
            .float() / 255.0
        )

        mask = torch.from_numpy(
            (mask >= 128).astype(np.float32)
        ).unsqueeze(0)

        return image_a, image_b, mask