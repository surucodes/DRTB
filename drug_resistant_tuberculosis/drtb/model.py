"""Model training and persistence helpers."""
from sklearn.naive_bayes import GaussianNB
import joblib
from typing import Any


def train_gaussian_nb(X_train, y_train) -> Any:
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def score(model, X, y):
    return model.score(X, y)


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
