import multiprocessing as mp

import neurokit2 as nk
import numpy as np
from biosppy.signals import ecg
from imblearn import over_sampling, pipeline
from sklearn import ensemble, model_selection, preprocessing

from fcache import fcache


def main():
    np.random.seed(42)

    X_train, y_train, X_test = load_data()
    print(X_train.shape, y_train.shape, X_test.shape)

    X_train, X_test = map(preprocess, [X_train, X_test])
    print(X_train.shape, X_test.shape)

    model = pipeline.make_pipeline(
        over_sampling.RandomOverSampler(random_state=42),
        preprocessing.StandardScaler(),
        ensemble.HistGradientBoostingClassifier(l2_regularization=0.2),
    )
    score = model_selection.cross_val_score(model, X_train, y_train, cv=5, n_jobs=6)
    print(score.mean(), score.std())  # 0.7930434002321605 0.011642850729758493

    create_submission(model, X_train, y_train, X_test)


@fcache
def load_data():
    X_train = read_raw_data("data/X_train.csv")
    y_train = np.genfromtxt("data/y_train.csv", delimiter=",", skip_header=1)[:, 1:]
    X_test = read_raw_data("data/X_test.csv")
    y_train = y_train.ravel()
    return X_train, y_train, X_test


def read_raw_data(fname):
    data = []
    with open(fname, "r") as file:
        for line in file.readlines()[1:]:
            data.append(np.fromstring(line, sep=",")[1:])
    return np.array(data, dtype=object)


@fcache
def preprocess(X):
    X = extract_features(X)
    X = create_features(X)
    return X


@fcache
def extract_features(X):
    Xn = []
    for x in X:
        _, clean, rpeaks, _, epochs, _, rate = ecg.ecg(x, 300, show=False)
        signals, info = nk.ecg_delineate(clean, rpeaks, 300)
        Xn.append((clean, rpeaks, epochs, rate, signals, info))
    return Xn


@fcache
def create_features(X):
    with mp.Pool(6) as pool:
        Xn = pool.map(_create_features, X)
    return np.array(Xn)


def _create_features(x):
    xn = []
    clean, R, epochs, rate, signals, info = x

    clip = clean[R[0] : R[-1]]
    freq = np.fft.rfftfreq(len(clip), 1 / 300)
    spec = np.abs(np.fft.rfft(clip)) / len(clip)
    freq, spec = binned(freq, spec, 50.0, 100, np.max)
    xn += list(spec)

    time = np.linspace(0, len(clip) / 300, len(clip))
    autocorr = np.correlate(clip, clip, mode="full") / len(clip)
    autocorr = autocorr[autocorr.size // 2 :]
    time, autocorr = binned(time, autocorr, 1.0, 100, np.mean)
    xn += list(autocorr)

    time = np.linspace(0, epochs.shape[1] / 300, epochs.shape[1])
    mean = np.nanmean(epochs, axis=0)
    std = np.nanstd(epochs, axis=0)
    _, mean = binned(time, mean, time[-1], 100, np.mean)
    time, std = binned(time, std, time[-1], 100, np.mean)
    xn += list(mean)
    xn += list(std)

    xn += msmv(rate)

    P = np.array(info["ECG_P_Peaks"])
    Q = np.array(info["ECG_Q_Peaks"])
    S = np.array(info["ECG_S_Peaks"])
    T = np.array(info["ECG_T_Peaks"])
    xn += msmv(np.diff(P))
    xn += msmv(np.diff(Q))
    xn += msmv(np.diff(R))
    xn += msmv(np.diff(S))
    xn += msmv(np.diff(T))

    xn += msmv(clean[P[~np.isnan(P)].astype(int)])
    xn += msmv(clean[Q[~np.isnan(Q)].astype(int)])
    xn += msmv(clean[R[~np.isnan(R)].astype(int)])
    xn += msmv(clean[S[~np.isnan(S)].astype(int)])
    xn += msmv(clean[T[~np.isnan(T)].astype(int)])

    PR_I = np.array(signals["ECG_R_Onsets"]) - np.array(signals["ECG_P_Onsets"])
    PR_S = np.array(signals["ECG_R_Onsets"]) - np.array(signals["ECG_P_Offsets"])
    QRS = np.array(signals["ECG_R_Offsets"]) - np.array(signals["ECG_R_Onsets"])
    ST_S = np.array(signals["ECG_T_Onsets"]) - np.array(signals["ECG_R_Offsets"])
    QT_I = np.array(signals["ECG_T_Offsets"]) - np.array(signals["ECG_R_Onsets"])
    xn += msmv(PR_I)
    xn += msmv(PR_S)
    xn += msmv(QRS)
    xn += msmv(ST_S)
    xn += msmv(QT_I)

    return xn


def binned(x, y, xend, nbins, func):
    bx = np.linspace(x[0], xend, nbins + 1)
    idx = np.digitize(x, bx, nbins) - 1
    bx = (bx[1:] + bx[:-1]) / 2
    by = np.array([func(y[idx == i]) for i in range(nbins)])
    return bx, by


def msmv(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [0, 0, 0, 0]
    if len(x) == 1:
        return [x[0], 0, x[0], 0]
    else:
        return [np.mean(x), np.std(x), np.median(x), np.var(x)]


def create_submission(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.vstack((np.arange(X_test.shape[0]), pred)).T
    np.savetxt(
        "submission.csv", pred, fmt="%.16g", delimiter=",", header="id,y", comments=""
    )


if __name__ == "__main__":
    main()
