"""
src/data/loader.py
------------------
Handles loading raw datasets from disk.
Works in both local (VS Code) and Google Colab environments.
"""

import pandas as pd
import os
import sys

# Allow importing config from project root when running as a module or from a notebook
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import cfg


def load_faa_full(nrows: int = None) -> pd.DataFrame:
    """
    Load the full FAA National Wildlife Strike Database (Public.xlsx).
    This is the primary dataset: 331,828 records × 101 columns.

    Parameters
    ----------
    nrows : int, optional
        Load only the first N rows (useful for quick testing).

    Returns
    -------
    pd.DataFrame
    """
    path = cfg.RAW_FAA_XLSX
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAA dataset not found at:\n  {path}\n\n"
            "Please copy Public.xlsx into the data/raw/ folder.\n"
            "If using Colab, upload it to your Drive under:\n"
            "  MyDrive/Flight-Bird-Strike/data/raw/Public.xlsx"
        )
    print(f"Loading FAA dataset from: {path}")
    df = pd.read_excel(path, nrows=nrows, engine="openpyxl")
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def load_bird_strikes_csv(nrows: int = None) -> pd.DataFrame:
    """
    Load the simplified Bird_strikes.csv (25,429 records × 26 columns).
    Useful for quick prototyping.

    Parameters
    ----------
    nrows : int, optional
        Load only the first N rows.

    Returns
    -------
    pd.DataFrame
    """
    path = cfg.RAW_BS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Bird_strikes.csv not found at:\n  {path}\n\n"
            "Please copy Bird_strikes.csv into the data/raw/ folder."
        )
    print(f"Loading Bird_strikes.csv from: {path}")
    df = pd.read_csv(path, nrows=nrows)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def load_processed() -> pd.DataFrame:
    """Load the preprocessed dataset saved after running the preprocessing notebook."""
    path = cfg.PROCESSED_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at:\n  {path}\n\n"
            "Please run notebook 02_Preprocessing.ipynb first."
        )
    df = pd.read_csv(path)
    print(f"Loaded processed dataset: {len(df):,} rows × {df.shape[1]} columns")
    return df
