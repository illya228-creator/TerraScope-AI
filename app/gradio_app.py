
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gradio_compat import configure_localhost_proxy_bypass


configure_localhost_proxy_bypass()

import gradio as gr
import numpy as np
from PIL import Image

from src.predictor import ChangePredictor


# ============================================================
# Paths and models
# ============================================================

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINTS = {
    "Controlled fine-tune": (
        CHECKPOINT_DIR / "siamese_unet_controlled_finetune_best.pth"
    ),
    "House experiment": (
        CHECKPOINT_DIR / "siamese_unet_house001_experiment.pth"
    ),
}


# ============================================================
# CSS
# ============================================================

CSS = """
:root {
    color-scheme: dark;
}

body,
.gradio-container {
    background: #0b1220 !important;
    color: #e8eef8 !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 24px !important;
}

.main-header {
    padding: 10px 0 22px;
}

.main-header h1 {
    font-size: clamp(30px, 4vw, 42px) !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
    color: #f4f7fc !important;
    margin: 0 0 8px !important;
}

.brand-accent {
    color: #58a0ff !important;
}

.subtitle {
    color: #b7c7dc !important;
    font-size: 16px !important;
    line-height: 1.6;
}

.panel {
    background: #111d2d !important;
    border: 1px solid #344963 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .12);
}

.panel-title {
    color: #f1f6ff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    margin-bottom: 6px;
}

.panel-description {
    color: #aebfd4 !important;
    font-size: 13px !important;
    line-height: 1.5;
    margin-bottom: 12px;
}

.upload-panel,
.settings-panel,
.result-panel {
    min-width: 0;
}

.upload-panel .wrap,
.upload-panel .image-container,
.result-panel .wrap,
.result-panel .image-container {
    border-radius: 10px !important;
}

.upload-panel .wrap {
    background: #0e1928 !important;
    border: 1px dashed #526d8d !important;
}

.upload-panel img,
.result-panel img {
    object-fit: contain !important;
}

.gradio-container label,
.gradio-container .label {
    color: #dce7f7 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #152337 !important;
    border-color: #344963 !important;
    color: #e8eef8 !important;
    border-radius: 8px !important;
}

.gradio-container input[type="range"] {
    accent-color: #2583ff !important;
}

.primary-action {
    margin: 10px 0 16px !important;
}

.primary-action button {
    width: 100% !important;
    min-height: 54px !important;
    border: none !important;
    border-radius: 11px !important;
    background: linear-gradient(100deg, #1680ff, #0861e9) !important;
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    box-shadow: 0 7px 20px rgba(13, 105, 245, .2);
}

.primary-action button:hover {
    filter: brightness(1.12);
}

.analysis-summary textarea {
    min-height: 110px !important;
    background: #152337 !important;
    border-color: #344963 !important;
    color: #e8eef8 !important;
}

footer {
    display: none !important;
}

@media (max-width: 700px) {
    .gradio-container {
        padding: 12px !important;
    }

    .panel {
        padding: 12px !important;
    }
}
"""


# ============================================================
# Prediction
# ============================================================

def predict_changes(before, after, model_name, threshold):
    if before is None or after is None:
        raise gr.Error("Загрузите оба спутниковых снимка.")

    checkpoint_path = CHECKPOINTS.get(model_name)

    if checkpoint_path is None:
        raise gr.Error("Выбрана неизвестная модель.")

    if not checkpoint_path.is_file():
        raise gr.Error(
            f"Не найден чекпоинт модели:\n{checkpoint_path}"
        )

    try:
        predictor = ChangePredictor(
            checkpoint_path=checkpoint_path,
            threshold=float(threshold),
        )

        result = predictor.predict(before, after)
        mask = result["mask"]

        after_resized = np.asarray(
            after.convert("RGB").resize(
                (1024, 1024),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        ).copy()

        # Overlay detected changes in red.
        overlay = after_resized.copy()
        changed = mask > 0
        red = np.array([255, 55, 65], dtype=np.float32)

        overlay[changed] = (
            0.45 * overlay[changed].astype(np.float32)
            + 0.55 * red
        ).astype(np.uint8)

        summary = (
            f"Изменено пикселей: "
            f"{result['changed_percentage']:.3f}%\n"
            f"Порог обнаружения: {result['threshold']:.2f}\n"
            f"Модель: {model_name}\n"
            f"Устройство: {predictor.device}\n"
            f"Размер анализа: 1024 × 1024"
        )

        return mask, overlay, summary

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(
            f"Не удалось выполнить анализ: {exc}"
        ) from exc


# ============================================================
# Interface
# ============================================================

with gr.Blocks(title="TerraScope AI") as demo:

    with gr.Column():

        gr.HTML(
            """
            <div class="main-header">
                <h1>
                    Terra<span class="brand-accent">Scope</span> AI
                </h1>
                <div class="subtitle">
                    Обнаружение изменений на спутниковых снимках
                    с помощью Siamese U-Net.
                </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):

            with gr.Column(
                scale=1,
                elem_classes=["panel", "upload-panel"],
            ):
                gr.HTML(
                    """
                    <div class="panel-title">
                        Снимок до строительства
                    </div>
                    <div class="panel-description">
                        Спутниковый снимок до предполагаемых изменений
                    </div>
                    """
                )

                before_image = gr.Image(
                    type="pil",
                    height=250,
                    show_label=False,
                    sources=["upload", "clipboard"],
                )

            with gr.Column(
                scale=1,
                elem_classes=["panel", "upload-panel"],
            ):
                gr.HTML(
                    """
                    <div class="panel-title">
                        Снимок после строительства
                    </div>
                    <div class="panel-description">
                        Спутниковый снимок после предполагаемых изменений
                    </div>
                    """
                )

                after_image = gr.Image(
                    type="pil",
                    height=250,
                    show_label=False,
                    sources=["upload", "clipboard"],
                )

        with gr.Row(equal_height=True):

            with gr.Column(
                scale=1,
                elem_classes=["panel", "settings-panel"],
            ):
                gr.HTML(
                    """
                    <div class="panel-title">Модель</div>
                    <div class="panel-description">
                        Выберите обученный чекпоинт
                    </div>
                    """
                )

                model_choice = gr.Dropdown(
                    choices=list(CHECKPOINTS.keys()),
                    value="Controlled fine-tune",
                    label="",
                    show_label=False,
                    interactive=True,
                )

            with gr.Column(
                scale=1,
                elem_classes=["panel", "settings-panel"],
            ):
                gr.HTML(
                    """
                    <div class="panel-title">
                        Порог обнаружения изменений
                    </div>
                    <div class="panel-description">
                        Низкий порог повышает чувствительность,
                        но может увеличить число ложных срабатываний.
                    </div>
                    """
                )

                threshold_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.30,
                    step=0.05,
                    label="Порог",
                    interactive=True,
                )

        run_button = gr.Button(
            "Обнаружить изменения",
            variant="primary",
            elem_classes=["primary-action"],
        )

        with gr.Row(equal_height=True):

            with gr.Column(
                scale=1,
                elem_classes=["panel", "result-panel"],
            ):
                gr.HTML(
                    """
                    <div class="panel-title">Маска изменений</div>
                    <div class="panel-description">
                        Белые области — пиксели, которые модель
                        классифицировала как изменения.
                    </div>
                    """
                )

                mask_output = gr.Image(
                    label="",
                    height=280,
                    show_label=False,
                    interactive=False,
                )

            with gr.Column(
                scale=1,
                elem_classes=["panel", "result-panel"],
            ):
                gr.HTML(
                    """
                    <div class="panel-title">
                        Изменения на снимке
                    </div>
                    <div class="panel-description">
                        Красным выделены области, обнаруженные моделью.
                    </div>
                    """
                )

                overlay_output = gr.Image(
                    label="",
                    height=280,
                    show_label=False,
                    interactive=False,
                )

        with gr.Column(elem_classes=["panel"]):

            gr.HTML(
                """
                <div class="panel-title">Результат анализа</div>
                <div class="panel-description">
                    Сводка последнего предсказания
                </div>
                """
            )

            summary_output = gr.Textbox(
                value=(
                    "Загрузите оба снимка и нажмите "
                    "«Обнаружить изменения»."
                ),
                show_label=False,
                interactive=False,
                lines=4,
                elem_classes=["analysis-summary"],
            )

    run_button.click(
        fn=predict_changes,
        inputs=[
            before_image,
            after_image,
            model_choice,
            threshold_slider,
        ],
        outputs=[
            mask_output,
            overlay_output,
            summary_output,
        ],
    )


# ============================================================
# Launch
# ============================================================

def main():
    """Start the local Gradio application."""
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CSS,
        max_file_size="20mb",
        inbrowser=True,
    )


if __name__ == "__main__":
    main()

