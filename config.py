"""
config.py
---------
Central configuration for the Flight-Bird Strike Risk Prediction project.
Works identically in VS Code (local) and Google Colab.

Usage:
    from config import cfg
    df = pd.read_excel(cfg.RAW_FAA_XLSX)
"""

import os
import sys

# ─── Environment Detection ───────────────────────────────────────────────────

def _in_colab() -> bool:
    return "google.colab" in sys.modules

IN_COLAB = _in_colab()


# ─── Path Configuration ───────────────────────────────────────────────────────

class Config:
    def __init__(self):
        if IN_COLAB:
            # Workflow:
            #   1. GitHub repo is cloned to /content/flight-bird-strike  (by setup cell)
            #   2. Drive is mounted at /content/drive                     (by setup cell)
            #   3. Public.xlsx + Bird_strikes.csv are copied from Drive   (by setup cell)
            #      source: MyDrive/Flight-Bird-Strike-MyResearch/data/raw/
            #      dest:   /content/flight-bird-strike/data/raw/
            # config.py then works identically to local — just pointing at the cloned repo.
            self.PROJECT_ROOT   = "/content/flight-bird-strike"
            self.DATA_ROOT      = "/content/flight-bird-strike/data"
            self.RAW_DATA_DIR   = "/content/flight-bird-strike/data/raw"

            # Path to your datasets on Google Drive (used by the setup cell only)
            self.DRIVE_RAW_DIR  = "/content/drive/MyDrive/Flight-Bird-Strike-MyResearch/data/raw"
        else:
            # Local Windows / VS Code
            # Automatically resolve paths relative to this config.py file
            self.PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
            self.DATA_ROOT      = os.path.join(self.PROJECT_ROOT, "data")
            self.RAW_DATA_DIR   = os.path.join(self.PROJECT_ROOT, "data", "raw")

        # ── Sub-directories (same structure everywhere) ──────────────────────
        self.PROCESSED_DIR  = os.path.join(self.DATA_ROOT, "processed")
        self.EXTERNAL_DIR   = os.path.join(self.DATA_ROOT, "external")
        self.MODELS_DIR     = os.path.join(self.PROJECT_ROOT, "saved_models")
        self.REPORTS_DIR    = os.path.join(self.PROJECT_ROOT, "reports")
        self.FIGURES_DIR    = os.path.join(self.REPORTS_DIR, "figures")
        self.RESULTS_DIR    = os.path.join(self.REPORTS_DIR, "results")

        # ── Raw dataset file paths ────────────────────────────────────────────
        self.RAW_FAA_XLSX   = os.path.join(self.RAW_DATA_DIR, "Public.xlsx")
        self.RAW_BS_CSV     = os.path.join(self.RAW_DATA_DIR, "Bird_strikes.csv")

        # ── Processed dataset file paths ─────────────────────────────────────
        self.PROCESSED_CSV  = os.path.join(self.PROCESSED_DIR, "faa_processed.csv")
        self.TRAIN_CSV      = os.path.join(self.PROCESSED_DIR, "train.csv")
        self.TEST_CSV       = os.path.join(self.PROCESSED_DIR, "test.csv")

        # ── Model save paths ─────────────────────────────────────────────────
        self.RF_MODEL_PATH  = os.path.join(self.MODELS_DIR, "random_forest.joblib")
        self.XGB_MODEL_PATH = os.path.join(self.MODELS_DIR, "xgboost.joblib")
        self.LGB_MODEL_PATH = os.path.join(self.MODELS_DIR, "lightgbm.joblib")

        # ── Modelling config ─────────────────────────────────────────────────
        self.TARGET_COLUMN      = "DAMAGE_LEVEL"    # Multi-class: None/Minor/Moderate/Substantial/Destroyed
        self.BINARY_TARGET      = "INDICATED_DAMAGE" # Binary: Yes/No
        self.RANDOM_STATE       = 42
        self.TEST_SIZE          = 0.20
        self.CV_FOLDS           = 5

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [self.PROCESSED_DIR, self.MODELS_DIR, self.FIGURES_DIR, self.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)

    def __repr__(self):
        env = "Google Colab" if IN_COLAB else "Local (VS Code)"
        return (
            f"Config(env={env!r},\n"
            f"  project_root={self.PROJECT_ROOT!r},\n"
            f"  raw_faa_xlsx={self.RAW_FAA_XLSX!r})"
        )


cfg = Config()
cfg.ensure_dirs()


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print(cfg)
    print(f"\nRunning in: {'Google Colab' if IN_COLAB else 'Local environment'}")
    print(f"FAA XLSX exists: {os.path.exists(cfg.RAW_FAA_XLSX)}")
    print(f"Bird strikes CSV exists: {os.path.exists(cfg.RAW_BS_CSV)}")
