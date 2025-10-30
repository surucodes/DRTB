"""Utility helpers: persistence and model evaluation."""
from typing import Dict, Any, Tuple
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from .logger import get_logger
from .exceptions import CustomException
import sys

logger = get_logger(__name__)


def save_object(file_path: str, obj: Any) -> None:
    try:
        # ensure parent directory exists
        from pathlib import Path
        p = Path(file_path).expanduser()
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, str(p))
        saved = str(p.resolve())
        logger.info(f"Saved object to {saved}")
        return saved
    except Exception as e:
        logger.exception("Failed to save object")
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    try:
        return joblib.load(file_path)
    except Exception as e:
        logger.exception("Failed to load object")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: Dict[str, Any], params: Dict[str, dict] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Train multiple candidate models and return accuracy report and fitted estimators.

    If params contains a parameter grid for a model name, GridSearchCV will be used.
    """
    report = {}
    fitted_models = {}
    params = params or {}

    # convert pandas inputs to numpy arrays to avoid feature-name issues with some libraries (e.g., xgboost)
    try:
        import pandas as _pd
        if isinstance(X_train, _pd.DataFrame):
            X_train_np = X_train.to_numpy()
            X_test_np = X_test.to_numpy()
        else:
            X_train_np = X_train
            X_test_np = X_test
        if hasattr(y_train, "to_numpy"):
            y_train_np = y_train.to_numpy()
            y_test_np = y_test.to_numpy()
        else:
            y_train_np = y_train
            y_test_np = y_test
    except Exception:
        X_train_np, X_test_np, y_train_np, y_test_np = X_train, X_test, y_train, y_test

    for name, model in models.items():
        try:
            logger.info(f"Training model: {name}")
            if name in params and params[name]:
                grid = GridSearchCV(model, params[name], cv=3, n_jobs=-1, scoring='accuracy')
                grid.fit(X_train_np, y_train_np)
                best = grid.best_estimator_
                logger.info(f"Best params for {name}: {grid.best_params_}")
            else:
                best = model
                best.fit(X_train_np, y_train_np)

            preds = best.predict(X_test_np)
            acc = accuracy_score(y_test, preds)
            report[name] = acc
            fitted_models[name] = best
            logger.info(f"Model {name} test accuracy: {acc:.4f}")
        except Exception as e:
            logger.exception(f"Failed training/eval for model {name}")
            report[name] = 0.0
            fitted_models[name] = None
    return report, fitted_models
