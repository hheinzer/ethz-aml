import numpy as np
import pandas as pd
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


def main():
    np.random.seed(42)

    X_train, y_train, X_test = load_data()
    print(X_train.shape, y_train.shape, X_test.shape)

    X_train, y_train = remove_outliers(X_train, y_train)
    X_train, X_test = preprocess(X_train, X_test)
    X_train, X_test = select_features(X_train, y_train, X_test)
    print(X_train.shape, y_train.shape, X_test.shape)

    model = pipeline.make_pipeline(
        # linear_model.LinearRegression(),
        # svm.SVR(),
        # ensemble.GradientBoostingRegressor(),
        # ensemble.ExtraTreesRegressor(),
        ensemble.StackingRegressor(
            estimators=[
                ("svr", svm.SVR(C=60.0, epsilon=1e-05)),
                ("gbm", ensemble.GradientBoostingRegressor(learning_rate=0.095)),
                ("etr", ensemble.ExtraTreesRegressor()),
            ],
            final_estimator=linear_model.Ridge(),
        )  # 0.6865593406215309
    )
    param_grid = {
        # "svr__C": np.linspace(10, 100, 10),
        # "svr__epsilon": np.logspace(-8, -4, 9),
        # "gradientboostingregressor__learning_rate": np.linspace(0.07, 0.11, 9),
    }
    search = model_selection.GridSearchCV(model, param_grid, cv=5, n_jobs=6)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(search.best_score_, search.best_params_)

    create_submission(model, X_train, y_train, X_test)


def load_data():
    X_train = pd.read_csv("data/X_train.csv", index_col="id")
    y_train = pd.read_csv("data/y_train.csv", index_col="id")
    X_test = pd.read_csv("data/X_test.csv", index_col="id")
    y_train = y_train["y"].to_numpy()
    return X_train, y_train, X_test


def remove_outliers(X_train, y_test):
    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        impute.SimpleImputer(strategy="median"),
        decomposition.PCA(n_components=2),
        ensemble.IsolationForest(contamination=0.05),
    )
    pred = model.fit_predict(X_train)
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
    np.savetxt("submission.csv", pred, delimiter=",", header="id,y", comments="")


if __name__ == "__main__":
    main()
