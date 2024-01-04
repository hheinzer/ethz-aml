# Task 2

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
