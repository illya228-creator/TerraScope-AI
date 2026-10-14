# TerraScope-AI

## Русский

**Обнаружение изменений на спутниковых изображениях с помощью глубокого обучения**

TerraScope AI — исследовательский проект в области компьютерного зрения и дистанционного зондирования Земли. Система анализирует два спутниковых изображения одной территории, полученных в разные моменты времени, и выявляет изменения застройки.

### Цели проекта

- Реализовать воспроизводимый конвейер обнаружения изменений.
- Обучать и оценивать модели глубокого обучения с использованием PyTorch.
- Сравнить как минимум две конфигурации моделей.
- Оценивать предсказания по метрикам Precision, Recall, F1-score и IoU.
- Анализировать ложноположительные и ложноотрицательные предсказания.
- Создать интерактивную демонстрацию на Gradio.

### Датасет

Основной датасет — LEVIR-CD, содержащий пары спутниковых изображений и бинарные маски изменений.

Доступ к датасету, структура каталогов и условия использования будут проверены на этапе подготовки данных.

### Технологии

- Python
- PyTorch
- NumPy
- Pillow
- Matplotlib
- Google Colab
- Gradio
- Git и GitHub
- pytest

### Статус проекта

**Этап 1 — настройка репозитория и проверка датасета.**

Метрики качества моделей будут добавлены после проведения реальных экспериментов. На текущем этапе проект не заявляет результаты бенчмарков.

### Планируемая структура

```text
TerraScope-AI/
├── app/
├── configs/
├── docs/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   └── training/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Воспроизводимость

По мере развития проекта будут документироваться конфигурации обучения, случайные seed, процедуры оценки и результаты экспериментов.

### Локальная демонстрация

Запустите интерактивную демонстрацию из корня репозитория:

```powershell
.\.venv\Scripts\python.exe .\app\gradio_app.py
```

Локальная среда использует системный прокси. Приложение явно обходит его для `127.0.0.1`, `localhost` и `::1`, чтобы проверка запуска Gradio могла обратиться к локальному серверу.

### Ограничения

Система представляет собой учебный исследовательский прототип. Предсказания зависят от датасета, качества изображений, пространственного совмещения и характеристик обучающих данных.

---

## English

**Satellite Image Change Detection with Deep Learning**

TerraScope AI is a research project in computer vision and remote sensing. The system analyzes two satellite images of the same area captured at different times and identifies changes in built-up areas.

### Project Goals

- Implement a reproducible change-detection pipeline.
- Train and evaluate deep-learning models using PyTorch.
- Compare at least two model configurations.
- Evaluate predictions with Precision, Recall, F1-score, and IoU.
- Analyze false-positive and false-negative predictions.
- Build an interactive demonstration using Gradio.

### Dataset

The primary dataset is LEVIR-CD, which contains pairs of satellite images and binary change masks.

Dataset access, directory structure, and usage conditions will be verified during the data-preparation stage.

### Tech Stack

- Python
- PyTorch
- NumPy
- Pillow
- Matplotlib
- Google Colab
- Gradio
- Git and GitHub
- pytest

### Project Status

**Stage 1 — repository setup and dataset validation.**

Model performance metrics will be added after actual experiments are conducted. No benchmark results are claimed at this stage.

### Planned Structure

```text
TerraScope-AI/
├── app/
├── configs/
├── docs/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   └── training/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Reproducibility

Training configurations, random seeds, evaluation procedures, and experiment results will be documented as the project develops.

### Local Demo

Run the interactive demo from the repository root:

```powershell
.\.venv\Scripts\python.exe .\app\gradio_app.py
```

The local environment uses a system proxy. The application explicitly bypasses it for `127.0.0.1`, `localhost`, and `::1` so Gradio's launch self-check can reach the locally started server.

### Limitations

This system is an educational research prototype. Predictions depend on the dataset, image quality, spatial alignment, and the characteristics of the training data.
