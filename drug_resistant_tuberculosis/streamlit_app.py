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
            st.subheader("Best model details")
            st.write(str(best_model))
            if save_model_opt:
                from drtb.model import save_model
                save_model(best_model, model_path)
                st.write(f"Saved best model to `{model_path}`")


if __name__ == "__main__":
    # allow `python streamlit_app.py` to launch the app for convenience
    run()
