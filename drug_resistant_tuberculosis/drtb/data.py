"""Data loading utilities for the DR-TB project."""
from typing import Optional
import os
import pandas as pd


def load_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the DR dataset CSV.

    If csv_path is None, assumes `dr_dataset.csv` is next to this package root.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(base_dir, "dr_dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df
