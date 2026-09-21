# Experiment 4: Binary Classification using Linear and Kernel-Based Models

Student: Isha Venkatataman
Register Number: 3122247001023
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To classify emails as spam or ham using Logistic Regression and SVM, and study how hyperparameter tuning (regularization strength, kernel choice) affects classification performance.

## Datasets
- UCI Spambase dataset: 4,601 emails, 57 numeric features, binary classification.

## Models Evaluated
1. Logistic Regression
2. Support Vector Machine (SVM)

## Methodological Workflow
- Load, preprocess, and perform EDA on Spambase data.
- Train baseline Logistic Regression and tune hyperparameters.
- Train baseline SVM with all kernels and tune SVM.
- Evaluate and compare models using 5-fold cross-validation.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook "Experiment4_Binary_Classification.ipynb"
```
