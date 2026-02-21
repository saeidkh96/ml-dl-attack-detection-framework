# scripts/config.py

from pathlib import Path

# scripts/ is inside repo_root/scripts/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

MERGED_FILE = DATA_PROCESSED / "merged_attacks.csv"

DL_PREPROC_PATH = RESULTS_DIR / "dl_preproc.joblib"
LSTM_PATH = RESULTS_DIR / "model_dl_lstm.keras"
CNN_PATH  = RESULTS_DIR / "model_dl_cnn.keras"

OUT_PRED = RESULTS_DIR / "splunk_model_predictions.csv"
OUT_SUMM = RESULTS_DIR / "splunk_model_summary.csv"

RANDOM_STATE = 42
