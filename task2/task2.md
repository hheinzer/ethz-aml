# Task 2

1. feature extraction
    - use biosppy to extract raw features
    - find R, S, Q, P, and T points with manual algorithm
    - use binned FFT, autocorrelation, and wavelets of full spectrum
    - use mean, variance, and standard deviation of the features combined
2. preprocessing
    - because the dataset is imbalanced, we use random over sampling
    - scale every feature to zero mean and unit variance
3. training
    - use GradientBoostingClassifier from sklearn
    - optimize Hyperparameters with RandomizedGridSearchCV
