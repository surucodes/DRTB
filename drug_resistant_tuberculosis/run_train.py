"""Small runner script to execute the full training pipeline.

Usage: python run_train.py
"""
from drtb import load_data, preprocess_pipeline, split_X_y
from drtb.preprocess import apply_smote, train_test_split_stratified
from drtb.model import train_and_select_best, save_model
from drtb.metrics import print_confusion_matrix, convert_binary_category_to_string
from sklearn.metrics import confusion_matrix, classification_report
import os
from drtb.logger import get_logger
from drtb.exceptions import CustomException

logger = get_logger(__name__)


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "dr_dataset.csv")
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    df_proc = preprocess_pipeline(df)
    X, y = split_X_y(df_proc)
    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_res, y_res)

    try:
        # candidate classifiers
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        try:
            from catboost import CatBoostClassifier
            has_catboost = True
        except Exception:
            has_catboost = False

        models = {
            "RandomForest": RandomForestClassifier(n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "KNeighbors": KNeighborsClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        if has_catboost:
            models["CatBoost"] = CatBoostClassifier(verbose=False)

        # small hyperparameter grids to try (kept small for speed)
        params = {
            "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
            "GradientBoosting": {"n_estimators": [50], "learning_rate": [0.1]},
            "XGBoost": {"n_estimators": [50], "learning_rate": [0.1]},
            "KNeighbors": {"n_neighbors": [3, 5]}
        }

        best_name, best_model, report = train_and_select_best(X_train, y_train, X_test, y_test, models, params)
        logger.info(f"Model performance report: {report}")
        print(f"Best model: {best_name} with accuracy {report[best_name]:.4f}")

        # show metrics for best model
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print_confusion_matrix(cm, ["Drug Resistant TB (DR)", "Drug Sensitive TB (DS)"])
        print(classification_report(convert_binary_category_to_string(y_test), convert_binary_category_to_string(y_pred)))

        model_path = os.path.join(base_dir, f'{best_name}_model.joblib')
        save_model(best_model, model_path)
        print(f"Saved best model to {model_path}")
    except CustomException as ce:
        logger.exception("Training pipeline failed with a custom exception")
        raise
    except Exception as e:
        logger.exception("Training pipeline failed")
        raise


if __name__ == '__main__':
    main()
