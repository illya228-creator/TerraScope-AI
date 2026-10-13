
from pathlib import Path

from src.predictor import ChangePredictor


ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "siamese_unet_controlled_finetune_best.pth"
)


def main():
    print("1. Проверка чекпоинта...")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    print(f"Чекпоинт найден: {CHECKPOINT.name}")

    print("\n2. Загрузка модели...")
    predictor = ChangePredictor(
        checkpoint_path=CHECKPOINT,
        threshold=0.30,
    )

    print(f"Модель загружена на устройство: {predictor.device}")

    print("\n3. Тестовый прогон на двух одинаковых изображениях...")
    import numpy as np

    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    result = predictor.predict(image, image)

    mask = result["mask"]

    print(f"Размер маски: {mask.shape}")
    print(f"Тип маски: {mask.dtype}")
    print(f"Уникальные значения: {np.unique(mask)}")
    print(
        "Процент пикселей, классифицированных как изменения: "
        f"{result['changed_percentage']:.4f}%"
    )

    assert mask.shape == (1024, 1024), "Неверный размер маски"
    assert mask.dtype == np.uint8, "Неверный тип маски"
    assert set(np.unique(mask)).issubset({0, 255}), (
        "Маска должна содержать только 0 и 255"
    )

    print("\nУСПЕХ: модель загрузилась, предсказание выполнено.")


if __name__ == "__main__":
    main()
