# scripts/score_all_models_for_splunk.py

from __future__ import annotations

import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import (
    MERGED_FILE,
    RESULTS_DIR,
    OUT_PRED,
    OUT_SUMM,
    DL_PREPROC_PATH,
    LSTM_PATH,
    CNN_PATH,
    RANDOM_STATE,
)

# Ensure results directory exists

RESULTS_DIR.mkdir(parents=True, exist_ok=True)



# Helpers (MUST match training)

def normalize_labels(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.replace({
        "3": "ddos_udp",
        "7": "remote_code_execution",
        "dos_csvvvv": "dos",
        "sql injection": "sql_injection",
        "sql-injection2": "sql_injection",
        "sqlinjection-updated": "sql_injection",
    })


def make_groups(df: pd.DataFrame) -> pd.Series:
    # Must match training logic
    if "TCP Stream" in df.columns:
        print("[INFO] Using GroupShuffleSplit groups = TCP Stream")
        return df["TCP Stream"].fillna(-1).astype(int)

    has_ip = ("IP Source" in df.columns) and ("IP Destination" in df.columns)
    if has_ip and ("TCP Source Port" in df.columns) and ("TCP Destination Port" in df.columns):
        print("[INFO] Using GroupShuffleSplit groups = IP pair + ports")
        return (
            df["IP Source"].astype(str)
            + "->"
            + df["IP Destination"].astype(str)
            + ":"
            + df["TCP Source Port"].astype(str)
            + "->"
            + df["TCP Destination Port"].astype(str)
        )

    if has_ip:
        print("[INFO] Using GroupShuffleSplit groups = IP pair")
        return df["IP Source"].astype(str) + "->" + df["IP Destination"].astype(str)

    raise ValueError(
        "No suitable grouping columns found for group split. "
        "Need at least 'TCP Stream' OR ('IP Source' and 'IP Destination')."
    )


def pick_time_column(df_like: pd.DataFrame) -> str | None:
    for col in ["Frame Time (Epoch)", "_time", "Frame Time"]:
        if col in df_like.columns:
            return col
    return None



# Load data

df = pd.read_csv(MERGED_FILE, low_memory=False)
if "label" not in df.columns:
    raise ValueError("Column 'label' not found in dataset")

# keep consistent with training drops 
DROP_COLS = ["No.", "Info", "source_file", "Frame Time", "Frame Time (Epoch)"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

# same label normalization as training
y_raw = normalize_labels(df["label"])
X = df.drop(columns=["label"], errors="ignore")

# Drop all-NaN columns 
all_nan_cols = [c for c in X.columns if X[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN columns:", all_nan_cols)
    X = X.drop(columns=all_nan_cols)

# reuse EXACT label order from training (saved in DL_PREPROC_PATH)
if not DL_PREPROC_PATH.exists():
    raise RuntimeError(
        f"Missing {DL_PREPROC_PATH}. "
        "Run train_model.py first (it saves label_classes to dl_preproc.joblib)."
    )

dl = joblib.load(DL_PREPROC_PATH)
label_classes = np.array(dl["label_classes"], dtype=object)

le = LabelEncoder()
le.classes_ = label_classes  # <-- critical: do NOT fit again

# Transform with training class order
y = le.transform(y_raw)

print("[INFO] Label classes (from training):", le.classes_)
print("[INFO] Label counts (normalized):\n", pd.Series(y_raw).value_counts())

# match training split method (GroupShuffleSplit)
groups = make_groups(df)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y[train_idx]
y_test = y[test_idx]

y_raw_train = y_raw.iloc[train_idx].copy()
y_raw_test = y_raw.iloc[test_idx].copy()

print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)

# Time column for Splunk output
time_col = pick_time_column(X_test)
if time_col:
    print("Using time column for Splunk output:", time_col)
else:
    print("No time column found. Output will not include event_time.")


pred_rows: list[pd.DataFrame] = []
summary_rows: list[dict] = []



# A) Score classic models (.joblib)

classic_model_paths = sorted(glob.glob(str(RESULTS_DIR / "model_*.joblib")))

if not classic_model_paths:
    print(" No classic models found (results/model_*.joblib).")

for mp in classic_model_paths:
    model_name = Path(mp).stem.replace("model_", "")
    print(f"\nScoring CLASSIC model: {model_name}")

    model = joblib.load(mp)

    y_pred = model.predict(X_test)  # predictions are encoded ints
    pred_label = le.inverse_transform(y_pred)  # decode with training order

    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        conf = np.max(proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    precw = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recw = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    summary_rows.append({
        "model": model_name,
        "family": "classic",
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "precision_weighted": float(precw),
        "recall_weighted": float(recw),
        "n_test": int(len(y_test)),
    })

    base = pd.DataFrame({
        "model": model_name,
        "family": "classic",
        "true_label": y_raw_test.values,
        "pred_label": pred_label,
    })

    if conf is not None:
        base["pred_confidence"] = conf

    if time_col:
        base["event_time"] = X_test[time_col].values

    pred_rows.append(base)



# B) Score deep learning models (.keras)

num_cols_dl = dl["num_cols_dl"]
imputer = dl["imputer"]
mean = dl["mean"]
std = dl["std"]

# Build numeric test features exactly like training
missing_cols = [c for c in num_cols_dl if c not in X_test.columns]
if missing_cols:
    raise RuntimeError(
        "Some numeric DL columns are missing from X_test.\n"
        f"Missing: {missing_cols[:20]}{' ...' if len(missing_cols) > 20 else ''}\n"
        "This usually means the dataset schema changed between training and scoring."
    )

Xte_num = X_test[num_cols_dl].copy()
Xte_num = imputer.transform(Xte_num)
Xte_num = (Xte_num - mean) / std

n_features = Xte_num.shape[1]
Xte_seq = Xte_num.reshape((Xte_num.shape[0], n_features, 1))

# ---- LSTM
if LSTM_PATH.exists():
    print("\nScoring DEEP model: dl_lstm")
    lstm = tf.keras.models.load_model(LSTM_PATH)

    proba = lstm.predict(Xte_seq, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    pred_label = le.inverse_transform(y_pred)
    conf = np.max(proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    precw = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recw = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    summary_rows.append({
        "model": "dl_lstm",
        "family": "deep",
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "precision_weighted": float(precw),
        "recall_weighted": float(recw),
        "n_test": int(len(y_test)),
    })

    base = pd.DataFrame({
        "model": "dl_lstm",
        "family": "deep",
        "true_label": y_raw_test.values,
        "pred_label": pred_label,
        "pred_confidence": conf,
    })

    if time_col:
        base["event_time"] = X_test[time_col].values

    pred_rows.append(base)
else:
    print(" LSTM model not found:", str(LSTM_PATH))

# ---- CNN
if CNN_PATH.exists():
    print("\nScoring DEEP model: dl_cnn")
    cnn = tf.keras.models.load_model(CNN_PATH)

    proba = cnn.predict(Xte_seq, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    pred_label = le.inverse_transform(y_pred)
    conf = np.max(proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    precw = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recw = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    summary_rows.append({
        "model": "dl_cnn",
        "family": "deep",
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "precision_weighted": float(precw),
        "recall_weighted": float(recw),
        "n_test": int(len(y_test)),
    })

    base = pd.DataFrame({
        "model": "dl_cnn",
        "family": "deep",
        "true_label": y_raw_test.values,
        "pred_label": pred_label,
        "pred_confidence": conf,
    })

    if time_col:
        base["event_time"] = X_test[time_col].values

    pred_rows.append(base)
else:
    print(" CNN model not found:", str(CNN_PATH))



# Save outputs for Splunk

if not pred_rows:
    raise RuntimeError("No predictions generated. Check that models exist in results/.")

pred_df = pd.concat(pred_rows, ignore_index=True)
summ_df = pd.DataFrame(summary_rows).sort_values("f1_weighted", ascending=False)

pred_df.to_csv(OUT_PRED, index=False)
summ_df.to_csv(OUT_SUMM, index=False)

print("\nSaved predictions:", str(OUT_PRED))
print("Saved summary    :", str(OUT_SUMM))
print("\nTop models by weighted F1:")
print(summ_df.head(10).to_string(index=False))
