"""
src/data/preprocessor.py
------------------------
Full preprocessing pipeline for the FAA Wildlife Strike Database.

Pipeline steps:
    1. Select and rename relevant columns
    2. Parse dates and extract temporal features
    3. Handle missing values
    4. Encode the target variable (DAMAGE_LEVEL)
    5. Encode categorical features
    6. Drop leakage / free-text columns
    7. Save processed data to disk
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import cfg


# ─── Column selection ────────────────────────────────────────────────────────

# Columns we actually use from the 101-column Public.xlsx
SELECTED_COLUMNS = [
    # Target
    "DAMAGE_LEVEL",
    "INDICATED_DAMAGE",
    # Temporal
    "INCIDENT_MONTH",
    "INCIDENT_YEAR",
    "TIME_OF_DAY",
    # Location
    "STATE",
    "FAAREGION",
    "AIRPORT_LATITUDE",
    "AIRPORT_LONGITUDE",
    # Flight
    "PHASE_OF_FLIGHT",
    "HEIGHT",
    "SPEED",
    "AC_CLASS",
    "AC_MASS",
    "TYPE_ENG",
    "NUM_ENGS",
    # Weather
    "SKY",
    "PRECIPITATION",
    # Wildlife
    "SPECIES",
    "SIZE",
    "NUM_SEEN",
    "NUM_STRUCK",
    # Operational
    "WARNED",
    "NR_INJURIES",
    "NR_FATALITIES",
]

# ─── Target encoding ─────────────────────────────────────────────────────────

DAMAGE_LEVEL_MAP = {
    "N": 0,   # No damage
    "M": 1,   # Minor
    "M?": 1,  # Minor (uncertain)
    "S": 2,   # Substantial
    "D": 3,   # Destroyed
}

DAMAGE_LEVEL_LABELS = {0: "None", 1: "Minor", 2: "Substantial", 3: "Destroyed"}

# ─── Main preprocessing function ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline on the raw FAA DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded by loader.load_faa_full()
    save : bool
        If True, saves the result to cfg.PROCESSED_CSV

    Returns
    -------
    pd.DataFrame  –  clean, model-ready DataFrame
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input shape: {df.shape}")

    # ── Step 1: Select columns ────────────────────────────────────────────────
    cols_present = [c for c in SELECTED_COLUMNS if c in df.columns]
    df = df[cols_present].copy()
    print(f"\n[1] Selected {len(cols_present)} columns -> shape {df.shape}")

    # ── Step 2: Drop rows where target is missing ─────────────────────────────
    before = len(df)
    df = df.dropna(subset=["DAMAGE_LEVEL"])
    print(f"[2] Dropped {before - len(df):,} rows with missing DAMAGE_LEVEL -> {len(df):,} rows")

    # ── Step 3: Encode target variable ───────────────────────────────────────
    df["damage_label"] = df["DAMAGE_LEVEL"].map(DAMAGE_LEVEL_MAP)
    # Drop unmapped (unexpected) values
    before = len(df)
    df = df.dropna(subset=["damage_label"])
    df["damage_label"] = df["damage_label"].astype(int)
    print(f"[3] Target encoded: {df['damage_label'].value_counts().to_dict()}")
    if before - len(df) > 0:
        print(f"    Dropped {before - len(df)} rows with unknown DAMAGE_LEVEL values")

    # ── Step 4: Binary target ─────────────────────────────────────────────────
    if "INDICATED_DAMAGE" in df.columns:
        # Column may be boolean (True/False) or string ('Y'/'N')
        ind = df["INDICATED_DAMAGE"]
        if ind.dtype == object:
            df["damage_binary"] = (ind.astype(str).str.upper() == "Y").astype(int)
        else:
            df["damage_binary"] = ind.fillna(False).astype(bool).astype(int)
    else:
        df["damage_binary"] = (df["damage_label"] > 0).astype(int)
    print(f"[4] Binary target: {df['damage_binary'].value_counts().to_dict()}")

    # ── Step 5: Temporal features ─────────────────────────────────────────────
    df["month"] = pd.to_numeric(df.get("INCIDENT_MONTH"), errors="coerce")
    df["year"]  = pd.to_numeric(df.get("INCIDENT_YEAR"), errors="coerce")
    df["season"] = df["month"].map(_month_to_season).fillna("Unknown")
    print(f"[5] Temporal features created: month, year, season")

    # ── Step 6: Handle missing values ─────────────────────────────────────────
    numeric_cols = ["HEIGHT", "SPEED", "NUM_SEEN", "NUM_STRUCK",
                    "NR_INJURIES", "NR_FATALITIES", "NUM_ENGS",
                    "AIRPORT_LATITUDE", "AIRPORT_LONGITUDE", "AC_MASS"]
    for col in numeric_cols:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            df[col] = numeric_series.fillna(median_val)

    cat_cols = ["PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "TIME_OF_DAY",
                "AC_CLASS", "TYPE_ENG", "STATE", "FAAREGION", "WARNED", "SIZE"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Species: group rare species into "Other"
    if "SPECIES" in df.columns:
        df["SPECIES"] = df["SPECIES"].fillna("Unknown")
        top_species = df["SPECIES"].value_counts().head(50).index
        df["species_grouped"] = df["SPECIES"].where(df["SPECIES"].isin(top_species), "Other")
    print(f"[6] Missing values handled")

    # ── Step 7: Encode categoricals ───────────────────────────────────────────
    cat_encode_cols = ["PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "TIME_OF_DAY",
                       "AC_CLASS", "TYPE_ENG", "STATE", "FAAREGION", "WARNED",
                       "SIZE", "season", "species_grouped"]
    df_encoded = pd.get_dummies(df, columns=[c for c in cat_encode_cols if c in df.columns],
                                drop_first=False, dtype=int)
    print(f"[7] One-hot encoding applied -> shape {df_encoded.shape}")

    # ── Step 8: Drop original raw columns we no longer need ──────────────────
    drop_cols = ["DAMAGE_LEVEL", "INDICATED_DAMAGE", "INCIDENT_MONTH",
                 "INCIDENT_YEAR", "SPECIES"]
    df_encoded = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])

    # ── Step 9: Final cleanup ─────────────────────────────────────────────────
    df_encoded = df_encoded.dropna()
    print(f"[8] Final shape after cleanup: {df_encoded.shape}")
    print(f"\nTarget distribution (damage_label):")
    vc = df_encoded["damage_label"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"   {DAMAGE_LEVEL_LABELS.get(k, k)} ({k}): {v:,}  ({100*v/len(df_encoded):.1f}%)")

    # ── Step 10: Save ─────────────────────────────────────────────────────────
    if save:
        os.makedirs(os.path.dirname(cfg.PROCESSED_CSV), exist_ok=True)
        df_encoded.to_csv(cfg.PROCESSED_CSV, index=False)
        print(f"\nSaved to: {cfg.PROCESSED_CSV}")

    print("=" * 60)
    return df_encoded


def _month_to_season(month):
    if pd.isna(month):
        return "Unknown"
    month = int(month)
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    return "Unknown"


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all feature columns (everything except target columns)."""
    exclude = {"damage_label", "damage_binary"}
    return [c for c in df.columns if c not in exclude]
