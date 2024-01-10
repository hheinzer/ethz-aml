import numpy as np
from sklearn import (
    decomposition,
    ensemble,
    feature_selection,
    impute,
    linear_model,
    model_selection,
    pipeline,
    preprocessing,
    svm,
)

from explore import plot_outliers


def main():
    X_train, y_train, X_test = load_data()
    print(X_train.shape, y_train.shape, X_test.shape)

    X_train, y_train = remove_outliers(X_train, y_train, X_test)
    X_train, X_test = preprocess(X_train, X_test)
    X_train, X_test = select_features(X_train, y_train, X_test)
    print(X_train.shape, y_train.shape, X_test.shape)

    model = pipeline.make_pipeline(
        ensemble.StackingRegressor(
            estimators=[
                ("svr", svm.SVR(C=60.0, epsilon=1e-05)),
                ("gbm", ensemble.GradientBoostingRegressor(learning_rate=0.095)),
                ("etr", ensemble.ExtraTreesRegressor()),
            ],
            final_estimator=linear_model.Ridge(),
        )
    )
    score = model_selection.cross_val_score(model, X_train, y_train, cv=5, n_jobs=6)
    print(score.mean(), score.std())  # 0.6844646263431688 0.02663668357699777

    create_submission(model, X_train, y_train, X_test)


def load_data():
    X_train = np.genfromtxt("data/X_train.csv", delimiter=",", skip_header=1)[:, 1:]
    y_train = np.genfromtxt("data/y_train.csv", delimiter=",", skip_header=1)[:, 1:]
    X_test = np.genfromtxt("data/X_test.csv", delimiter=",", skip_header=1)[:, 1:]
    y_train = y_train.ravel()
    return X_train, y_train, X_test


def remove_outliers(X_train, y_test, X_test):
    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        impute.SimpleImputer(strategy="median"),
        decomposition.PCA(n_components=2),
        ensemble.IsolationForest(contamination=0.045),
    )
    pred = model.fit_predict(X_train)
    plot_outliers(model[:3].transform(X_train), model[:3].transform(X_test), pred)
    X_train, y_test = X_train[pred > 0], y_test[pred > 0]
    return X_train, y_test


def preprocess(X_train, X_test):
    model = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        impute.SimpleImputer(strategy="median"),
    )
    X_train = model.fit_transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test


def select_features(X_train, y_train, X_test):
    model = pipeline.make_pipeline(
        feature_selection.VarianceThreshold(),
        feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=200),
        feature_selection.SelectFromModel(linear_model.Lasso(0.1)),
    )
    model.fit(X_train, y_train)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test


def create_submission(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.vstack((np.arange(X_test.shape[0]), pred)).T
    np.savetxt("submission.csv", pred, fmt="%.16g", delimiter=",", header="id,y", comments="")


if __name__ == "__main__":
    main()
