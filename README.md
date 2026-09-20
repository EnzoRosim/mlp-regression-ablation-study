# MLP Regression with Ablation Study

A PyTorch implementation of a Multi-Layer Perceptron (MLP) for a 1D regression task, comparing a vanilla SGD baseline against models with momentum, L1/L2 regularization, and dropout. Developed as Project 1 for the *Introduction to Artificial Neural Networks* course.

## Overview

The goal is to study how different training components affect generalization under a deliberately constrained data regime (10% train / 10% validation / 80% test). Starting from a plain SGD baseline (no momentum, no regularization), each ablation isolates the effect of a single component, followed by a combined model using the two best-performing techniques.

| Model | MSE | RMSE | MAE | R² | Epochs to Stop |
|---|---|---|---|---|---|
| Baseline (vanilla SGD) | 0.5049 | 0.7106 | 0.5858 | 0.0236 | 150 |
| + Momentum | 0.5022 | 0.7087 | 0.5755 | 0.0288 | 150 |
| + L2 | 0.5002 | 0.7073 | 0.5816 | 0.0326 | 150 |
| + L1 | 0.4880 | 0.6985 | 0.5759 | 0.0563 | 150 |
| + Dropout | 0.4936 | 0.7026 | 0.5789 | 0.0454 | 207 |
| **Combined** | **0.4823** | **0.6945** | **0.5620** | **0.0674** | **185** |

Full methodology, discussion, and figures are available in the [report](relatorio/relatorio.pdf).

## Project Structure

```
proj1-mlp-regression/
├── data/
│   └── dataset_projeto1.csv       # input dataset (x, y)
├── src/
│   ├── dataset.py                  # data loading, splitting, normalization
│   ├── model.py                    # MLP architecture
│   ├── metrics.py                  # MAE, MSE, RMSE, R²
│   ├── train.py                    # training loop, experiments, evaluation
│   └── busca_baseline.py           # empirical search for baseline hyperparameters
├── outputs/
│   ├── splits/                     # saved train/val/test indices (.npy)
│   ├── graficos/                   # generated training curves and diagnostic plots
│   └── logs/                       # per-experiment results (.json)
├── relatorio/
│   ├── relatorio.tex
│   └── relatorio.pdf
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12+
- PyTorch
- pandas
- scikit-learn
- matplotlib
- numpy

Install with:

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate the data split** (run once — indices are reused by every experiment):

```bash
python src/dataset.py
```

**2. (Optional) Run the baseline hyperparameter search:**

```bash
python src/busca_baseline.py
```

**3. Train the baseline and all ablations, and evaluate on the test set:**

```bash
python src/train.py
```

This trains six models (baseline, momentum, L2, L1, dropout, combined), saves a training curve and a JSON log for each under `outputs/`, and prints a comparison table to the console.

## Methodology Summary

- **Baseline:** plain SGD (no momentum, no regularization), architecture and learning rate selected via empirical search over the validation set.
- **Ablations:** each variation modifies exactly one component relative to the baseline (momentum, L2 weight decay, L1 penalty, or dropout), with architecture held fixed.
- **Reproducibility:** fixed random seed (42), train/val/test split persisted to disk and reused across all experiments, input/target standardization fit exclusively on the training set.
- **Evaluation:** final metrics computed on the held-out test set (80% of the data), never used during training or model selection.

## Report

The full report (in Portuguese) discussing the experimental setup, training dynamics, ablation results, and residual analysis is available at [`relatorio/relatorio.pdf`](relatorio/relatorio.pdf).
