"""Preprocessing helpers: encoding, one-hot, SMOTE, and splitting."""
from typing import Tuple
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
"""
Note: Import SMOTE lazily inside apply_smote to avoid import-time failures
when imbalanced-learn is missing or incompatible in environments that use
only prediction (no resampling needed).
"""


def _encode_yes_no(df: pd.DataFrame, yes_no_columns: list) -> pd.DataFrame:
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].replace({"Yes": 1, "No": 0})
    return df


def _encode_gender_and_class(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Female": 1, "Male": 0})
    if "Class" in df.columns:
        df["Class"] = df["Class"].replace({"DR": 1, "DS": 0})
    return df


def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(data=df, columns=[c for c in columns if c in df.columns])


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's preprocessing steps and return processed df."""
    df = df.copy()
    yes_no_columns = ["Contact DR", "Smoking", "Alcohol", "Cavitary pulmonary", "Diabetes", "TBoutside", "Class"]
    df = _encode_yes_no(df, yes_no_columns)
    df = _encode_gender_and_class(df)
    # One-hot Age and Nutritional as in notebook
    df = one_hot_encode(df, ["Age", "Nutritional"]) 
    return df


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop("Class", axis="columns")
    y = df["Class"]
    return X, y


def apply_smote(X, y, sampling_strategy: str = "minority"):
    # Lazy import to prevent module import failures during prediction-only flows
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def train_test_split_stratified(X, y, test_size=0.2, random_state=15):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
