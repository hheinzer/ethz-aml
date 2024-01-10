# Advanced Machine Learning Projects

Codes for the ETH Zurich course [Advanced Machine Learning](https://ml2.inf.ethz.ch/courses/aml/).

# Task 1: Predict the age of a brain from MRI features

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

# Task 2: Heart rhythm classification from raw ECG signals

1. feature extraction
    - use biosppy to extract raw features (cleaned signal, rpeaks, heart beats, hear rate)
    - find S, Q, P, and T points using neurokit
    - some of the signals are inverted, to combat this we add the inverse of all signals
    - use binned FFT and autocorrelation of full spectrum
    - compute various time intervals between R, S, Q, P, and T points (and their on/offsets)
    - use mean, standard deviation, median, and variance of the features
2. preprocessing
    - because the data set is imbalanced, we use random over sampling
    - scale every feature to zero mean and unit variance
3. training
    - use HistGradientBoostingClassifier from sklearn
    - optimize hyper parameters with RandomizedGridSearchCV

# Task 3: Mitral valve segmentation
