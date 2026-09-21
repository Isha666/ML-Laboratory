# Experiment 8: Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering

Student: Isha Venkataraman
Register Number: 3122247001023
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To study the clustering of smartphone sensor data using three unsupervised algorithms: K-Means, DBSCAN and Hierarchical Agglomerative Clustering (HAC). The assignment chooses the number of clusters with the elbow and silhouette methods, tunes the DBSCAN parameters (eps and min_samples), compares linkage criteria for hierarchical clustering, and evaluates all models with internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external metrics (Adjusted Rand Index, Normalized Mutual Information) against the true activity labels.

## Datasets
- Human Activity Recognition Using Smartphones (UCI HAR): 10,299 windows (7,352 train and 2,947 test windows merged), 561 time and frequency domain features, 6 activity classes, 30 volunteers (reduced to 10 principal components, 70.21% variance explained, for clustering)

## Models Evaluated
1. K-Means Clustering (k = 6 selected using the elbow method and silhouette analysis, tested for k = 2 to 8)
2. DBSCAN (eps = 6.0, min_samples = 20 selected from a tuning grid)
3. Hierarchical Agglomerative Clustering (Ward linkage, 6 clusters; single, complete and average linkage compared)

## Methodological Workflow
- Feature Preprocessing: Standardization using StandardScaler on all 561 features. No missing values were found and the activity labels are used only for evaluation.
- Dimensionality Reduction: Principal Component Analysis (PCA) to 10 components for clustering, and t-SNE (perplexity 30) for 2D visualization.
- Hyperparameter Tuning: Elbow and silhouette curves for k in K-Means, and a grid search over eps (2.5 to 7.0) and min_samples (5, 10, 20) for DBSCAN, with the choice made without using the true labels.
- Cluster Evaluation: Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index as internal metrics, Adjusted Rand Index and Normalized Mutual Information as external metrics, plus confusion matrices obtained by mapping clusters to activities with the Hungarian algorithm.
- Comparative Analysis: Bar plots of the metrics, dendrograms, per-activity recall and cluster plots in PCA and t-SNE space for the three algorithms.

## Installation and Execution
Download the UCI HAR dataset and place the extracted `UCI HAR Dataset` folder in the same directory as the notebook.

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-8.ipynb
```
