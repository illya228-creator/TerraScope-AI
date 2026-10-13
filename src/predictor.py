
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.siamese_unet import SiameseUNet


class ChangePredictor:
    """Predict changes between two satellite images."""

    def __init__(
        self,
        checkpoint_path,
        threshold=0.30,
        device=None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        # Support both a saved state_dict and a training checkpoint.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            base_channels = checkpoint.get("base_channels", 16)
            self.threshold = checkpoint.get("threshold", threshold)
        else:
            state_dict = checkpoint
            base_channels = 16

        self.model = SiameseUNet(
            base_channels=base_channels
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _prepare_image(image):
        """Convert an image to RGB and resize it to 1024x1024."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Image must be a file path, PIL image, or NumPy array."
            )

        image = image.convert("RGB")
        image = image.resize((1024, 1024), Image.Resampling.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)

        return tensor.unsqueeze(0)

    @torch.inference_mode()
    def predict(self, before, after):
        """
        Return the binary change mask and the percentage of changed pixels.

        Inputs can be file paths, PIL images, or NumPy arrays.
        """
        before_tensor = self._prepare_image(before).to(self.device)
        after_tensor = self._prepare_image(after).to(self.device)

        logits = self.model(before_tensor, after_tensor)
        probabilities = torch.sigmoid(logits)

        mask = (probabilities >= self.threshold).squeeze().cpu().numpy()
        mask = (mask.astype(np.uint8) * 255)

        changed_percentage = float(np.mean(mask > 0) * 100.0)

        return {
            "mask": mask,
            "changed_percentage": changed_percentage,
            "threshold": float(self.threshold),
        }
