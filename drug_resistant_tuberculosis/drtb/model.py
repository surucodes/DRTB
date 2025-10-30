"""Model training and persistence helpers."""
from sklearn.naive_bayes import GaussianNB
import joblib
from typing import Any, Dict
from .utils import evaluate_models, save_object
from .logger import get_logger
from .exceptions import CustomException
import sys

logger = get_logger(__name__)


def train_gaussian_nb(X_train, y_train) -> Any:
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def score(model, X, y):
    return model.score(X, y)


def save_model(model, path: str):
    """Save model object to disk and return absolute path."""
    return save_object(path, model)


def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.exception("Failed to load model")
        raise CustomException(e, sys)


def train_and_select_best(X_train, y_train, X_test, y_test, models: Dict[str, Any], params: Dict[str, dict] = None):
    """Train multiple candidate models and return the best model (by accuracy) and the report.

    Returns: (best_model_name, best_model, report_dict)
    """
    report, fitted = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    if not report:
        raise CustomException("No models evaluated", sys)
    best_score = max(report.values())
    best_name = [name for name, sc in report.items() if sc == best_score][0]
    best_model = fitted.get(best_name)
    logger.info(f"Best model selected: {best_name} with accuracy {best_score:.4f}")
    return best_name, best_model, report

