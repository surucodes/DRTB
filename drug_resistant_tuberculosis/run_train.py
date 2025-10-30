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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Path to input CSV. If not provided, prefers 'de_dataset.csv' then falls back to 'dr_dataset.csv' in the project folder.")
    args = parser.parse_args()

    # Prefer the older dataset 'de_dataset.csv' if present per user's request;
    # otherwise fall back to the repository's 'dr_dataset.csv'. Allow CLI override.
    default_de = os.path.join(base_dir, 'de_dataset.csv')
    default_dr = os.path.join(base_dir, 'dr_dataset.csv')
    if args.input:
        csv_path = args.input
    elif os.path.exists(default_de):
        csv_path = default_de
    else:
        csv_path = default_dr

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
        from sklearn.ensemble import VotingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.naive_bayes import GaussianNB, ComplementNB
        from sklearn.svm import SVC
        try:
            from catboost import CatBoostClassifier
            has_catboost = True
        except Exception:
            has_catboost = False
        try:
            from lightgbm import LGBMClassifier
            has_lgb = True
        except Exception:
            has_lgb = False

        from sklearn.ensemble import HistGradientBoostingClassifier

        from sklearn.ensemble import StackingClassifier

        models = {
            "RandomForest": RandomForestClassifier(n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "KNeighbors": KNeighborsClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "GaussianNB": GaussianNB(),
            "ComplementNB": ComplementNB(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "HistGradientBoosting": HistGradientBoostingClassifier(),
            "SVC": SVC(probability=True)
        }
        if has_lgb:
            models["LightGBM"] = LGBMClassifier()
        # Add a stacking ensemble of top candidate models (no hyperparameter tuning here)
        try:
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50))
            ]
            try:
                from catboost import CatBoostClassifier
                estimators.append(("cat", CatBoostClassifier(verbose=False, random_state=42)))
            except Exception:
                pass
            stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
            models["StackingEnsemble"] = stacking
        except Exception:
            # if stacking cannot be constructed, continue without it
            pass
        if has_catboost:
            models["CatBoost"] = CatBoostClassifier(verbose=False)

        # --- New ensemble models requested by user ---
        # Ensemble 1: RandomForest + GradientBoosting + DecisionTree (soft voting)
        try:
            v1_estimators = [
                ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=100)),
                ("dt", DecisionTreeClassifier())
            ]
            ensemble1 = VotingClassifier(estimators=v1_estimators, voting='soft', n_jobs=-1)
            models["Ensemble_RF_GB_DT"] = ensemble1
        except Exception:
            pass

        # Ensemble 2: XGBoost + (existing StackingEnsemble if available) + CatBoost
        try:
            v2_estimators = []
            # XGBoost
            try:
                xgb_est = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50)
                v2_estimators.append(("xgb", xgb_est))
            except Exception:
                pass

            # include stacking if it was successfully created above
            if 'stacking' in locals():
                try:
                    v2_estimators.append(("stacking", stacking))
                except Exception:
                    pass
            # CatBoost if available
            if has_catboost:
                try:
                    cat_est = CatBoostClassifier(verbose=False, random_state=42)
                    v2_estimators.append(("cat", cat_est))
                except Exception:
                    pass

            if v2_estimators:
                ensemble2 = VotingClassifier(estimators=v2_estimators, voting='soft', n_jobs=-1)
                models["Ensemble_XGB_Stack_Cat"] = ensemble2
        except Exception:
            pass

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
