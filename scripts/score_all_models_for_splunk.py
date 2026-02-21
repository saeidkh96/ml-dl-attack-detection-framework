# scripts/score_all_models_for_splunk.py

from __future__ import annotations

import glob
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
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

# ===============================
# Ensure results directory exists
# ===============================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Load data
# ===============================
df = pd.read_csv(MERGED_FILE, low_memory=False)
if "label" not in df.columns:
    raise ValueError("Column 'label' not found in dataset")

y_raw = df["label"].astype(str)
X = df.drop(columns=["label"])

# Drop all-NaN columns (same rule as training)
all_nan_cols = [c for c in X.columns if X[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN columns:", all_nan_cols)
    X = X.drop(columns=all_nan_cols)

# Encode labels (same mapping as training)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Fixed split (must match training for fair comparison)
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# Pick ONE time column to keep in output (optional)
def pick_time_column(df_like: pd.DataFrame) -> str | None:
    for col in ["Frame Time (Epoch)", "_time", "Frame Time"]:
        if col in df_like.columns:
            return col
    return None

time_col = pick_time_column(X_test)
if time_col:
    print("Using time column for Splunk output:", time_col)
else:
    print("No time column found. Output will not include event_time.")

pred_rows: list[pd.DataFrame] = []
summary_rows: list[dict] = []

# =========================================================
# A) Score classic models (.joblib)
# =========================================================
classic_model_paths = sorted(glob.glob(str(RESULTS_DIR / "model_*.joblib")))

if not classic_model_paths:
    print("No classic models found (results/model_*.joblib).")

for mp in classic_model_paths:
    model_name = mp.split("/")[-1].split("\\")[-1].replace(".joblib", "").replace("model_", "")
    print(f"\nScoring CLASSIC model: {model_name}")

    model = joblib.load(mp)

    y_pred = model.predict(X_test)
    pred_label = le.inverse_transform(y_pred)

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

# =========================================================
# B) Score deep learning models (.keras)
# =========================================================
if DL_PREPROC_PATH.exists():
    dl = joblib.load(DL_PREPROC_PATH)

    num_cols_dl = dl["num_cols_dl"]
    imputer = dl["imputer"]
    mean = dl["mean"]
    std = dl["std"]

    # Build numeric test features exactly like training
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
        print("CNN model not found:", str(CNN_PATH))

else:
    print("DL preprocessing file not found:", str(DL_PREPROC_PATH))
    print("Train deep models first (train_model.py).")

# =========================================================
# Save outputs for Splunk
# =========================================================
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
