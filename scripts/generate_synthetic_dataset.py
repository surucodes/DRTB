#!/usr/bin/env python3
"""
Generate a synthetic balanced dataset using SMOTENC.

This script will:
 - load an existing dataset at repo root named 'dr_dataset.csv' (or a path provided with --input)
 - clean a couple of known typos in the 'Age' column
 - label-encode categorical features with OrdinalEncoder
 - run SMOTENC to create synthetic samples up to `n_per_class` for each class
 - inverse-transform the encoded features and save the resulting CSV

Usage:
 python scripts/generate_synthetic_dataset.py --n_per_class 2500 --output outputs/synthetic_tb_dataset_5000.csv

Note: this script expects an input CSV with a 'Class' column containing labels (e.g. 'DR' and 'DS').
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC


def load_input_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}. Please provide a CSV (default 'dr_dataset.csv') or set --input to a path.")
    df = pd.read_csv(p)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Fix a couple of known typos sometimes present in raw text
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).replace({'>= 4Suchas': '>= 45 years', '>= 4B years': '>= 45 years'})
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='dr_dataset.csv', help="Path to input CSV (default: 'dr_dataset.csv')")
    parser.add_argument('--n_per_class', '-n', type=int, default=2500, help='Target samples per class after resampling')
    parser.add_argument('--output', '-o', default='outputs/synthetic_tb_dataset_5000.csv', help='Output CSV path')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_input_df(args.input)
    print(f"Loaded input with shape: {df.shape}")
    df = clean_df(df)
    print(f"After cleaning shape: {df.shape}")

    if 'Class' not in df.columns:
        raise SystemExit("Input must contain a 'Class' column with target labels (e.g. 'DR' and 'DS').")

    print("Original class distribution:")
    print(df['Class'].value_counts())

    X = df.drop(columns=['Class'])
    y = df['Class']

    feature_names = X.columns.tolist()

    # Ordinal encode features for SMOTENC
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X.astype(str))

    # all columns categorical here (we encoded everything). SMOTENC requires at
    # least one continuous feature; if none exists we fall back to SMOTEN
    # (SMOTE for nominal) or RandomOverSampler.
    categorical_features = list(range(X_enc.shape[1]))

    # build sampling dict: for each unique class, set target n_per_class
    classes = sorted(pd.Series(y.unique()).astype(str).tolist())
    sampling_strategy = {c: args.n_per_class for c in classes}

    # Detect if original data had any numeric columns -- if not, use SMOTEN
    # (works for categorical-only data). If SMOTEN isn't available, fall back
    # to RandomOverSampler which simply duplicates minority samples.
    has_numeric = any([pd.api.types.is_numeric_dtype(df[col]) for col in X.columns])

    if not has_numeric:
        try:
            from imblearn.over_sampling import SMOTEN

            print(f"No numeric features detected — using SMOTEN with targets: {sampling_strategy}")
            sm = SMOTEN(sampling_strategy=sampling_strategy, random_state=args.random_state)
            X_res_enc, y_res = sm.fit_resample(X_enc, y)
        except Exception:
            # final fallback
            from imblearn.over_sampling import RandomOverSampler

            print("SMOTEN not available or failed — falling back to RandomOverSampler (may duplicate rows).")
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=args.random_state)
            X_res_enc, y_res = ros.fit_resample(X_enc, y)
    else:
        print(f"Applying SMOTENC with sampling targets: {sampling_strategy}")
        sm = SMOTENC(categorical_features=categorical_features, sampling_strategy=sampling_strategy, random_state=args.random_state)
        X_res_enc, y_res = sm.fit_resample(X_enc, y)

    # inverse transform encoded features back to labels
    X_res = enc.inverse_transform(X_res_enc)

    df_res = pd.DataFrame(X_res, columns=feature_names)
    df_res['Class'] = y_res

    df_res.to_csv(out_path, index=False)

    print(f"Saved resampled dataset to {out_path} with shape {df_res.shape}")
    print("New class distribution:")
    print(df_res['Class'].value_counts())


if __name__ == '__main__':
    main()
