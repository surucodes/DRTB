"""Streamlit application for the DR-TB Gaussian Naive Bayes pipeline.

This file is safe to import (it avoids importing streamlit at module import time).
Call `run()` or run with `streamlit run streamlit_app.py`.
"""
from typing import Optional
import os

# Import project modules (these don't import streamlit)
from drtb.data import load_data
from drtb.preprocess import preprocess_pipeline, split_X_y, apply_smote, train_test_split_stratified
from drtb.model import train_gaussian_nb, save_model
from drtb.metrics import print_confusion_matrix, convert_binary_category_to_string, plot_roc
from sklearn.metrics import confusion_matrix, classification_report


def _train_and_evaluate(df, save_model_path: Optional[str] = None):
    """Train the model on the provided dataframe and return metrics and figures."""
    df_proc = preprocess_pipeline(df)
    X, y = split_X_y(df_proc)
    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_res, y_res)

    model = train_gaussian_nb(X_train, y_train)
    acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(convert_binary_category_to_string(y_test), convert_binary_category_to_string(y_pred))

    # optionally save
    if save_model_path:
        save_model(model, save_model_path)

    return {
        "model": model,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run():
    """Start the Streamlit UI. This function imports streamlit lazily so that module import stays lightweight."""
    import streamlit as st
    st.set_page_config(page_title="DR-TB Naive Bayes", layout="wide")

    st.title("Drug-Resistant Tuberculosis — Gaussian Naive Bayes")

    st.sidebar.header("Data & Options")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    use_default = st.sidebar.checkbox("Use repository dataset (dr_dataset.csv)", value=True)

    save_model_opt = st.sidebar.checkbox("Save trained model", value=True)
    model_path = st.sidebar.text_input("Model filename", value="gaussian_nb_model.joblib")

    if uploaded is not None:
        try:
            df = load_data(uploaded)
        except Exception:
            # streamlit's UploadedFile behaves like a file-like object for pandas
            df = load_data() if use_default else None
    else:
        if use_default:
            base_dir = os.path.dirname(__file__)
            csv_path = os.path.join(base_dir, "dr_dataset.csv")
            try:
                df = load_data(csv_path)
            except Exception as e:
                st.error(f"Could not load default dataset: {e}")
                st.stop()
        else:
            st.info("Please upload a CSV or toggle 'Use repository dataset' in the sidebar.")
            st.stop()

    st.subheader("Dataset preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

    with st.expander("Show column types"):
        st.write(df.dtypes)

    st.subheader("Preprocessing & Training")
    mode = st.radio("Action", ["Train & Evaluate (single)", "Model selection (compare multiple)"], index=0)

    if mode == "Train & Evaluate (single)":
        if st.button("Run preprocessing + train + evaluate"):
            with st.spinner("Training model — this may take a few seconds"):
                results = _train_and_evaluate(df, save_model_path=model_path if save_model_opt else None)

            st.success(f"Training finished — test accuracy: {results['accuracy']:.4f}")
            st.subheader("Classification report")
            st.text(results['classification_report'])

            st.subheader("Confusion Matrix")
            fig_cm = print_confusion_matrix(results['confusion_matrix'], ["Drug Resistant TB (DR)", "Drug Sensitive TB (DS)"])
            st.pyplot(fig_cm)

            st.subheader("ROC Curve")
            # plot_roc draws directly into pyplot; call it and capture the auc
            auc_val = plot_roc(results['y_test'], results['y_pred'])
            st.write(f"AUC: {auc_val:.3f}")
            st.pyplot()

            if save_model_opt:
                st.write(f"Model saved to `{model_path}`")

    else:
        st.write("This will train and compare multiple candidate classifiers and show their test accuracy.")
        run_selection = st.button("Run model selection")
        if run_selection:
            with st.spinner("Running model selection — may take longer depending on models"):
                # replicate the same candidate models and params as the CLI runner
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
                    "HistGradientBoosting": __import__('sklearn').ensemble.HistGradientBoostingClassifier(),
                    "SVC": SVC(probability=True)
                }
                if has_lgb:
                    models["LightGBM"] = LGBMClassifier()

                # Add stacking ensemble
                try:
                    from sklearn.ensemble import StackingClassifier
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
                    pass
                if has_catboost:
                    models["CatBoost"] = CatBoostClassifier(verbose=False)

                # --- Add the two ensemble models to the candidate list ---
                try:
                    # Ensemble 1: RF + GB + DT (soft voting)
                    v1_estimators = [
                        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                        ("gb", GradientBoostingClassifier(n_estimators=100)),
                        ("dt", DecisionTreeClassifier())
                    ]
                    ensemble1 = VotingClassifier(estimators=v1_estimators, voting='soft', n_jobs=-1)
                    models["Ensemble_RF_GB_DT"] = ensemble1
                except Exception:
                    pass

                try:
                    # Ensemble 2: XGB + stacking (if available) + CatBoost
                    v2_estimators = []
                    try:
                        xgb_est = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50)
                        v2_estimators.append(("xgb", xgb_est))
                    except Exception:
                        pass
                    if 'stacking' in locals():
                        try:
                            v2_estimators.append(("stacking", stacking))
                        except Exception:
                            pass
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

                params = {
                    "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                    "GradientBoosting": {"n_estimators": [50], "learning_rate": [0.1]},
                    "XGBoost": {"n_estimators": [50], "learning_rate": [0.1]},
                    "KNeighbors": {"n_neighbors": [3, 5]}
                }

                # preprocessing pipeline reuse
                from drtb.preprocess import preprocess_pipeline, split_X_y, apply_smote, train_test_split_stratified

                df_proc = preprocess_pipeline(df)
                X, y = split_X_y(df_proc)
                X_res, y_res = apply_smote(X, y)
                X_train, X_test, y_train, y_test = train_test_split_stratified(X_res, y_res)

                from drtb.model import train_and_select_best
                best_name, best_model, report = train_and_select_best(X_train, y_train, X_test, y_test, models, params)
            st.success(f"Model selection finished — best: {best_name} (accuracy={report[best_name]:.4f})")
            st.subheader("All model accuracies")
            st.write(report)

            # Show ensemble-only comparison for quick inspection
            try:
                import pandas as _pd
                acc_series = _pd.Series(report)
                # ensembles include names with 'Ensemble' or 'Stacking'
                ensemble_keys = [k for k in acc_series.index if 'Ensemble' in k or 'ensemble' in k or 'Stacking' in k or k.startswith('Ensemble_')]
                if len(ensemble_keys) > 0:
                    st.subheader('Ensemble models comparison')
                    ens_df = acc_series.loc[ensemble_keys].sort_values(ascending=False).rename('accuracy').to_frame()
                    st.dataframe(ens_df.style.format({'accuracy': '{:.4f}'}))
                    # bar chart for quick visual
                    st.bar_chart(ens_df['accuracy'])
                else:
                    st.info('No ensemble models were constructed in this run.')
            except Exception:
                # non-fatal - continue
                pass

            # Detailed visualizations and metrics for the best model
            st.subheader("Detailed results for best model")
            # predict on X_test
            try:
                # Ensure X_test is available in local scope
                from drtb.preprocess import split_X_y
                # Use X_test, y_test from outer scope
                y_pred = best_model.predict(X_test)
            except Exception:
                st.warning("Could not compute predictions for best model")
                y_pred = None

            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
            import pandas as pd
            import numpy as np

            if y_pred is not None:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.4f}")

                # confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                from drtb.metrics import print_confusion_matrix
                fig_cm = print_confusion_matrix(cm, ["Drug Resistant TB (DR)", "Drug Sensitive TB (DS)"])
                st.pyplot(fig_cm)

                # classification report
                st.subheader("Classification Report")
                cls_rep = classification_report(y_test, y_pred, target_names=["DS","DR"], output_dict=True)
                cls_df = pd.DataFrame(cls_rep).transpose()
                st.dataframe(cls_df.round(4))

                # ROC and PR curves if possible
                from drtb.metrics import plot_roc, plot_precision_recall
                probs = None
                if hasattr(best_model, 'predict_proba'):
                    probs_all = best_model.predict_proba(X_test)
                    # assume positive class is 1
                    if probs_all.shape[1] == 2:
                        probs = probs_all[:, 1]
                elif hasattr(best_model, 'decision_function'):
                    try:
                        probs = best_model.decision_function(X_test)
                    except Exception:
                        probs = None

                if probs is not None:
                    st.subheader("ROC Curve")
                    fig_roc, roc_auc = plot_roc(y_test, probs)
                    st.pyplot(fig_roc)
                    st.write(f"AUC: {roc_auc:.4f}")

                    st.subheader("Precision-Recall Curve")
                    fig_pr, avg_prec = plot_precision_recall(y_test, probs)
                    st.pyplot(fig_pr)
                    st.write(f"Average precision (AP): {avg_prec:.4f}")
                else:
                    st.info("Probability scores not available for this model — ROC/PR not shown.")

                # Feature importances (if available)
                st.subheader("Feature importances / coefficients")
                try:
                    # get feature names from df_proc
                    feature_names = list(df_proc.drop('Class', axis=1).columns)
                except Exception:
                    feature_names = None

                import matplotlib.pyplot as plt
                if hasattr(best_model, 'feature_importances_') and feature_names is not None:
                    import numpy as _np
                    fi = best_model.feature_importances_
                    idx = _np.argsort(fi)[::-1][:20]
                    fig = plt.figure(figsize=(8, 6))
                    plt.barh([feature_names[i] for i in idx[::-1]], fi[idx[::-1]])
                    plt.title('Top feature importances')
                    st.pyplot(fig)
                elif hasattr(best_model, 'coef_') and feature_names is not None:
                    coef = best_model.coef_
                    # handle multiclass vs binary
                    if coef.ndim == 1:
                        vals = coef
                    else:
                        vals = coef[0]
                    idx = np.argsort(np.abs(vals))[::-1][:20]
                    fig = plt.figure(figsize=(8, 6))
                    plt.barh([feature_names[i] for i in idx[::-1]], vals[idx[::-1]])
                    plt.title('Top coefficients (by magnitude)')
                    st.pyplot(fig)
                else:
                    st.info('No feature importances or coefficients available for this model.')

                # Cross-validation scores for the best model
                st.subheader('Cross-validation (5-fold) scores for best model')
                from sklearn.model_selection import cross_val_score
                try:
                    # convert to numpy arrays to avoid library issues (e.g., xgboost expecting no special feature names)
                    if hasattr(X_res, 'to_numpy'):
                        X_res_cv = X_res.to_numpy()
                    else:
                        X_res_cv = X_res
                    if hasattr(y_res, 'to_numpy'):
                        y_res_cv = y_res.to_numpy()
                    else:
                        y_res_cv = y_res

                    cv_scores = cross_val_score(best_model, X_res_cv, y_res_cv, cv=5, scoring='accuracy', n_jobs=-1)
                    st.write('CV accuracy mean: ', float(np.mean(cv_scores)))
                    st.write('CV accuracies: ', [float(x) for x in cv_scores])
                except Exception as e:
                    st.info(f'Could not compute cross-validation scores: {e}')

            if save_model_opt:
                from drtb.model import save_model
                save_model(best_model, model_path)
                st.write(f"Saved best model to `{model_path}`")


if __name__ == "__main__":
    # allow `python streamlit_app.py` to launch the app for convenience
    run()
