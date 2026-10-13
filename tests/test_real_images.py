
from pathlib import Path

import numpy as np
from PIL import Image

from src.predictor import ChangePredictor


ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "siamese_unet_controlled_finetune_best.pth"
)

BEFORE = ROOT / "test_images" / "before.png"
AFTER = ROOT / "test_images" / "after.png"
OUTPUT_DIR = ROOT / "outputs"


def main():
    for path in (CHECKPOINT, BEFORE, AFTER):
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка модели...")
    predictor = ChangePredictor(
        checkpoint_path=CHECKPOINT,
        threshold=0.30,
    )

    print("Чтение спутниковых снимков...")
    before = Image.open(BEFORE).convert("RGB")
    after = Image.open(AFTER).convert("RGB")

    print(f"Исходный размер before: {before.size}")
    print(f"Исходный размер after:  {after.size}")

    result = predictor.predict(before, after)
    mask = result["mask"]

    mask_path = OUTPUT_DIR / "change_mask.png"
    Image.fromarray(mask).save(mask_path)

    # Green overlay marks pixels classified as changed.
    before_array = np.asarray(before.resize((1024, 1024))).copy()
    after_array = np.asarray(after.resize((1024, 1024))).copy()

    changed = mask > 0
    overlay = after_array.copy()

    green = np.zeros_like(overlay)
    green[:, :, 1] = 255

    overlay[changed] = (
        0.45 * overlay[changed] + 0.55 * green[changed]
    ).astype(np.uint8)

    Image.fromarray(before_array).save(
        OUTPUT_DIR / "before_resized.png"
    )
    Image.fromarray(after_array).save(
        OUTPUT_DIR / "after_resized.png"
    )
    Image.fromarray(overlay).save(
        OUTPUT_DIR / "change_overlay.png"
    )

    print("\nРезультат:")
    print(f"Устройство: {predictor.device}")
    print(f"Порог: {result['threshold']}")
    print(
        "Доля пикселей, отмеченных как изменения: "
        f"{result['changed_percentage']:.4f}%"
    )
    print(f"Маска: {mask_path}")
    print(f"Наложение: {OUTPUT_DIR / 'change_overlay.png'}")


if __name__ == "__main__":
    main()
