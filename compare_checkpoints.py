
from pathlib import Path

import numpy as np
from PIL import Image

from src.predictor import ChangePredictor


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
TEST_IMAGES = ROOT / "test_images"
OUTPUT = ROOT / "outputs" / "checkpoint_comparison"

BEFORE = TEST_IMAGES / "before.png"
AFTER = TEST_IMAGES / "after.png"

MODELS = {
    "controlled_finetune": (
        CHECKPOINTS / "siamese_unet_controlled_finetune_best.pth"
    ),
    "house001_experiment": (
        CHECKPOINTS / "siamese_unet_house001_experiment.pth"
    ),
}


def main():
    for path in (BEFORE, AFTER, *MODELS.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    before = Image.open(BEFORE).convert("RGB")
    after = Image.open(AFTER).convert("RGB")

    results = []

    for name, checkpoint in MODELS.items():
        print(f"\nПроверяем: {name}")

        predictor = ChangePredictor(
            checkpoint_path=checkpoint,
            threshold=0.30,
        )

        result = predictor.predict(before, after)
        mask = result["mask"]

        # Save binary mask.
        Image.fromarray(mask).save(OUTPUT / f"{name}_mask.png")

        # Overlay predicted changes in green on the after image.
        after_resized = after.resize((1024, 1024))
        overlay = np.asarray(after_resized).copy()
        changed = mask > 0

        overlay[changed] = (
            0.45 * overlay[changed]
            + 0.55 * np.array([0, 255, 0])
        ).astype(np.uint8)

        Image.fromarray(overlay).save(
            OUTPUT / f"{name}_overlay.png"
        )

        percentage = float(np.mean(changed) * 100)

        results.append((name, percentage))

        print(f"Устройство: {predictor.device}")
        print(f"Порог: {result['threshold']}")
        print(f"Пиксели изменений: {percentage:.4f}%")
        print(f"Маска: {OUTPUT / f'{name}_mask.png'}")
        print(f"Наложение: {OUTPUT / f'{name}_overlay.png'}")

    print("\n--- Сравнение ---")
    for name, percentage in results:
        print(f"{name}: {percentage:.4f}% пикселей изменений")

    print(f"\nВсе результаты сохранены в: {OUTPUT}")


if __name__ == "__main__":
    main()
