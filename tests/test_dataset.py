from pathlib import Path

import pytest
import torch

from src.data.dataset import LEVIRDataset


def test_dataset_returns_correct_shapes():
    root = Path("data/processed")

    if not root.exists():
        pytest.skip("Local dataset is not downloaded")

    dataset = LEVIRDataset(root, "train")
    image_a, image_b, mask = dataset[0]

    assert image_a.shape == (3, 1024, 1024)
    assert image_b.shape == (3, 1024, 1024)
    assert mask.shape == (1, 1024, 1024)

    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
