# Task 1

1. Outlier removal
    - Remove median and scale data according to interquartile range
    - Imputation for missing values using k-Nearest Neighbors
    - Principal component analysis, reduction to 2 components
    - Isolation Forest Algorithm, with contamination of 4.5% to detect/remove outliers (55)
2. Preprocessing + feature selection
    - Standardize features by removing the mean and scaling to unit variance
    - Imputation for missing values using k-Nearest Neighbors
    - Remove features that have zero variance
    - Select the 200 features that have the highest correlation with target
    - Select features based on importance weights of Lasso regression (74)
3. Model selection
    - Stacked regression model consisting of
        - Support Vector Regression
        - Histogram-based Gradient Boosting Regression Tree
        - Extra-trees regressor
        - Multi-layer Perceptron regressor
    - All hyperparameters are found/validated through 10-fold cross-validated grid search
