# Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

**Student:** Isha Venkataraman
**Register Number:** 3122247001023
**Degree:** M.Tech Integrated CSE
**Course:** ICS1512 - Machine Learning Algorithms Laboratory

## Objective

To implement a single-layer Perceptron Learning Algorithm (PLA) from scratch and a Multilayer Perceptron (MLP) trained with backpropagation, and to compare them on the English Handwritten Characters dataset. The MLP is tuned in stages (activation function, optimizer, learning rate, batch size, architecture, loss function, regularization and data augmentation), and both models are evaluated with accuracy, precision, recall, F1-score, confusion matrices, micro/macro ROC curves and training error vs epochs curves.

## Dataset

* English Handwritten Characters Dataset: 3,410 images (1200 x 900 RGB scans), 62 classes (0-9, A-Z, a-z), 55 images per class, no missing values
* Preprocessed to 32 x 32 grayscale and flattened to 1,024 features
* Stratified split: 2,294 train (37 per class), 434 validation (7 per class), 682 test (11 per class)

## Models Evaluated

1. Model A: Single-layer Perceptron Learning Algorithm (62 one-vs-rest units, step activation, pocket weights, implemented from scratch with NumPy)
2. Model B (untuned baseline): MLP with one hidden layer of 256 ReLU units, Adam (lr 0.001), batch size 64, cross-entropy
3. Model B (tuned): MLP with one hidden layer of 256 ReLU units, SGD with momentum 0.9 (lr 0.05), batch size 32, cross-entropy, weight decay 1e-4, early stopping (PyTorch)

## Methodological Workflow

* Image Preprocessing: Grayscale conversion, inversion, bounding-box crop with padding, resize to 32 x 32 and scaling to [0, 1].
* Hyperparameter Tuning: Seven one-factor-at-a-time stages, each configuration trained with three seeds and selected on mean validation accuracy.
* Regularization: Early stopping, dropout, L2 weight decay and data augmentation were compared to reduce overfitting.
* Cross-Validation: Stratified 5-fold cross-validation of the final MLP configuration on the training plus validation data.
* Evaluation: Accuracy, macro precision, recall and F1, micro/macro ROC-AUC, confusion matrices, per-class F1 and convergence curves on the untouched test set.

## Results (Test Set)

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| PLA | 0.6437 | 0.6395 | 0.9600 |
| MLP (untuned) | 0.7229 | 0.7218 | 0.9872 |
| MLP (tuned) | 0.7317 | 0.7305 | 0.9847 |

## Installation and Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `english.csv` and the `Img` folder (from the Kaggle dataset) in the same folder as the notebook, then run the notebook:

```bash
jupyter notebook expt-9.ipynb
```

Figures are saved to the `images/` folder.
