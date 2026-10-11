import torch

from src.models.siamese_unet import SiameseUNet


def test_model_output_shape():
    model = SiameseUNet(base_channels=8)
    model.eval()

    image_a = torch.rand(1, 3, 128, 128)
    image_b = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        logits = model(image_a, image_b)

    assert logits.shape == (1, 1, 128, 128)


def test_model_rejects_different_input_shapes():
    model = SiameseUNet(base_channels=8)

    image_a = torch.rand(1, 3, 128, 128)
    image_b = torch.rand(1, 3, 64, 64)

    try:
        model(image_a, image_b)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for mismatched inputs")
