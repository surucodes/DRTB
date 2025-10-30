"""Small runner script to execute the full training pipeline.

Usage: python run_train.py
"""
from drtb import load_data, preprocess_pipeline, split_X_y
from drtb.preprocess import apply_smote, train_test_split_stratified
from drtb.model import train_gaussian_nb, score, save_model
from drtb import print_confusion_matrix, convert_binary_category_to_string
from sklearn.metrics import confusion_matrix, classification_report
import os


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "dr_dataset.csv")
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    df_proc = preprocess_pipeline(df)
    X, y = split_X_y(df_proc)
    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_res, y_res)

    model = train_gaussian_nb(X_train, y_train)
    acc = score(model, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm, ["Drug Resistant TB (DR)", "Drug Sensitive TB (DS)"])
    print(classification_report(convert_binary_category_to_string(y_test), convert_binary_category_to_string(y_pred)))

    # Save model
    model_path = os.path.join(base_dir, 'gaussian_nb_model.joblib')
    save_model(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()
