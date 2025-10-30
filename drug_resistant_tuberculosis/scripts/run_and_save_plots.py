"""Run model selection and save detailed plots to outputs/ for paper use."""
import os
from drtb import load_data
from drtb.preprocess import preprocess_pipeline, split_X_y, apply_smote, train_test_split_stratified
from drtb.model import train_and_select_best
from drtb.metrics import print_confusion_matrix, plot_roc, plot_precision_recall
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_and_save():
    base = os.path.dirname(os.path.dirname(__file__))
    csv = os.path.join(base, 'dr_dataset.csv')
    out_dir = os.path.join(base, 'outputs')
    ensure_dir(out_dir)

    df = load_data(csv)
    df_proc = preprocess_pipeline(df)
    X, y = split_X_y(df_proc)
    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_res, y_res)

    # models same as run_train
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, HistGradientBoostingClassifier
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
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "HistGradientBoosting": HistGradientBoostingClassifier()
    }
    if has_catboost:
        models["CatBoost"] = CatBoostClassifier(verbose=False)

    # stacking
    try:
        estimators = [("rf", RandomForestClassifier(n_estimators=50, random_state=42)), ("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50))]
        if has_catboost:
            estimators.append(("cat", CatBoostClassifier(verbose=False, random_state=42)))
        stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
        models["StackingEnsemble"] = stacking
    except Exception:
        pass

    params = {
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
        "GradientBoosting": {"n_estimators": [50], "learning_rate": [0.1]},
        "XGBoost": {"n_estimators": [50], "learning_rate": [0.1]},
        "KNeighbors": {"n_neighbors": [3, 5]}
    }

    best_name, best_model, report = train_and_select_best(X_train, y_train, X_test, y_test, models, params)

    # predictions and probs
    try:
        if hasattr(best_model, 'predict_proba'):
            probs = best_model.predict_proba(X_test)
            if probs.shape[1] == 2:
                probs_pos = probs[:, 1]
            else:
                probs_pos = None
        else:
            probs_pos = None
    except Exception:
        probs_pos = None

    y_pred = best_model.predict(X_test)

    # save confusion matrix
    from sklearn.metrics import confusion_matrix as _confusion_matrix
    cm_arr = _confusion_matrix(y_test, y_pred)
    fig_cm = print_confusion_matrix(cm_arr, ["Drug Resistant TB (DR)", "Drug Sensitive TB (DS)"])
    fig_cm.savefig(os.path.join(out_dir, 'confusion_matrix.png'), bbox_inches='tight')

    # ROC and PR
    if probs_pos is not None:
        fig_roc, roc_auc = plot_roc(y_test, probs_pos)
        fig_roc.savefig(os.path.join(out_dir, 'roc_curve.png'), bbox_inches='tight')

        fig_pr, avg_prec = plot_precision_recall(y_test, probs_pos)
        fig_pr.savefig(os.path.join(out_dir, 'pr_curve.png'), bbox_inches='tight')

    # feature importances
    try:
        feature_names = list(df_proc.drop('Class', axis=1).columns)
    except Exception:
        feature_names = None

    import matplotlib.pyplot as plt
    if hasattr(best_model, 'feature_importances_') and feature_names is not None:
        fi = best_model.feature_importances_
        idx = __import__('numpy').argsort(fi)[::-1][:20]
        fig = plt.figure(figsize=(8, 6))
        plt.barh([feature_names[i] for i in idx[::-1]], fi[idx[::-1]])
        plt.title('Top feature importances')
        fig.savefig(os.path.join(out_dir, 'feature_importances.png'), bbox_inches='tight')

    # classification report save as csv
    from sklearn.metrics import classification_report
    import pandas as pd
    cls_rep = classification_report(y_test, y_pred, target_names=["DS","DR"], output_dict=True)
    cls_df = pd.DataFrame(cls_rep).transpose()
    cls_df.to_csv(os.path.join(out_dir, 'classification_report.csv'))

    # save model
    try:
        from drtb.model import save_model
        save_model(best_model, os.path.join(out_dir, f'{best_name}_model.joblib'))
    except Exception:
        pass

    # write a short summary file
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as fh:
        fh.write(f'Best model: {best_name}\n')
        fh.write('Model accuracies:\n')
        for k, v in report.items():
            fh.write(f'  {k}: {v:.4f}\n')
        if probs_pos is not None:
            fh.write(f'ROC AUC: {roc_auc:.4f}\n')
            fh.write(f'Average precision (AP): {avg_prec:.4f}\n')

    print('Saved outputs to', out_dir)


if __name__ == '__main__':
    run_and_save()
