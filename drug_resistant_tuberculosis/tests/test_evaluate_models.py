import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from drtb.utils import evaluate_models
from sklearn.ensemble import RandomForestClassifier


def test_evaluate_models_returns_report_and_models():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=4, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {"rf": RandomForestClassifier(n_estimators=10, random_state=42)}
    report, fitted = evaluate_models(X_train, y_train, X_test, y_test, models)

    assert isinstance(report, dict)
    assert "rf" in report
    assert report["rf"] >= 0.0 and report["rf"] <= 1.0
    assert isinstance(fitted, dict)
    assert "rf" in fitted
    assert fitted["rf"] is not None
