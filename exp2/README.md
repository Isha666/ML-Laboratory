# Experiment 2: Spam Classification using Naive Bayes and KNN

Student: Isha Venkatataman
Register Number: 3122247001023
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To build spam classifiers using three variants of Naive Bayes (Gaussian, Multinomial, Bernoulli) and KNN, compare their performance, study the effect of changing k in KNN, and compare KDTree and BallTree.

## Datasets
- Spambase dataset: 4,601 emails, 57 numeric features, binary classification.

## Models Evaluated
1. Gaussian Naive Bayes
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes
4. k-Nearest Neighbors (KNN)

## Methodological Workflow
- Load and preprocess Spambase dataset.
- Exploratory Data Analysis (EDA).
- Train and evaluate Naive Bayes models.
- Hyperparameter tuning for KNN using GridSearchCV and RandomizedSearchCV.
- 5-fold cross-validation and evaluation using accuracy, precision, recall, and F1-score.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook "Experiment2_Full_Spam_Classification_fixed (1).ipynb"
```
