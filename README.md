# TerraScope-AI

**Satellite Image Change Detection with Deep Learning**

TerraScope AI — исследовательский проект в области Computer Vision и Remote Sensing. Система анализирует два спутниковых изображения одной территории, полученных в разные моменты времени, и выявляет изменения застройки.

## Project Goals

- Implement a reproducible change detection pipeline.
- Train and evaluate deep learning models using PyTorch.
- Compare at least two model configurations.
- Evaluate predictions using Precision, Recall, F1-score and IoU.
- Analyze false-positive and false-negative predictions.
- Build an interactive demonstration using Gradio.

## Dataset

The primary dataset is LEVIR-CD, containing pairs of satellite images and binary change masks.

Dataset access, directory structure and usage conditions will be verified during the data preparation stage.

## Tech Stack

- Python
- PyTorch
- NumPy
- Pillow
- Matplotlib
- Google Colab
- Gradio
- Git and GitHub
- pytest

## Project Status

**Stage 1 — Repository setup and dataset validation.**

Model performance metrics will be added after actual experiments. No benchmark results are claimed at this stage.

## Planned Structure

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

## Reproducibility

Training configurations, random seeds, evaluation procedures and experiment results will be documented as development progresses.

## Limitations

The system is an educational research prototype. Predictions depend on the dataset, image quality, spatial alignment and the characteristics of the training data.
