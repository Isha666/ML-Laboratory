# Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)

Student: Isha Venkataraman
Register Number: 3122247001023
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To study the effect of dimensionality reduction using Principal Component Analysis (PCA) on the performance of ten machine learning classifiers. Each model is evaluated under both No-PCA and With-PCA conditions, with hyperparameter tuning and 5-fold cross-validation, on the Digits dataset. As an extension, the same workflow is repeated with the regressor equivalent of each model on the Diabetes dataset, to check whether the same PCA behaviour holds for a continuous target.

## Datasets
- Digits: 1,797 samples, 64 features (8x8 grayscale pixel intensities), 10 classes (reduced to 40 components, 95.16% variance explained)
- Diabetes (regression extension): 442 samples, 10 features, continuous disease-progression target (reduced to 8 components, 99.15% variance explained)

## Models Evaluated
1. Support Vector Machine (SVM)
2. Gaussian Naive Bayes
3. k-Nearest Neighbors (KNN)
4. Logistic Regression
5. Decision Tree
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. XGBoost
10. Stacking (Base learners: Decision Tree, KNN, SVM; Meta-learner: Logistic Regression)

Regressor equivalents (SVR, Bayesian Ridge, KNN Regressor, Linear Regression, Decision Tree Regressor, Random Forest Regressor, AdaBoost Regressor, Gradient Boosting Regressor, XGBoost Regressor, Stacking Regressor) are used for the Diabetes extension.

## Methodological Workflow
- Feature Preprocessing: Leakage-free standardization using StandardScaler fitted strictly on the training split.
- Dimensionality Reduction: Principal Component Analysis (PCA) fitted on the training partition only, using a 95% cumulative explained variance target.
- Hyperparameter Tuning: Grid search over defined parameter spaces evaluated with 5-fold cross-validation, run independently for No-PCA and With-PCA.
- Cross-Validation: Stratified 5-fold cross-validation (KFold for regression) recording fold-wise accuracy/R², macro F1, and overall means.
- Evaluation: Held-out test-set accuracy, macro F1, ROC-AUC and confusion matrix for the best classifier; R², RMSE and MAE for the best regressor.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook mlexp7.ipynb
```

Run all cells top to bottom. Figures are saved at 600 DPI into a `figures/` folder created next to the notebook, and the two 5-fold cross-validation result tables are also exported as CSV files there:
- `figures/table5_cv_results_classification.csv`
- `figures/table5_cv_results_regression.csv`

Hyperparameter tuning for the classification part takes roughly 25 minutes on a standard CPU; the regression extension finishes in under a minute.

## Repository
GitHub: https://github.com/Isha666/ML-Laboratory
