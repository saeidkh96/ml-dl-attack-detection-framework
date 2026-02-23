# scripts/train_model.py

from __future__ import annotations

from build_dataset import build_merged_dataset

import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from keras import layers, models

from config import (
    MERGED_FILE,
    RESULTS_DIR,
    DL_PREPROC_PATH,
    LSTM_PATH,
    CNN_PATH,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")


# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load data (auto-build if missing)
if not MERGED_FILE.exists():
    print("[INFO] merged_attacks.csv not found. Building dataset from raw CSV files...")
    build_merged_dataset(force=False)

df = pd.read_csv(MERGED_FILE, low_memory=False)

if "label" not in df.columns:
    raise ValueError(
        "Column 'label' not found in dataset.\n"
        "Make sure your raw CSV files either contain a label column "
        "or are separated per attack type (one class per file)."
    )


# 0) Drop leak-prone / non-generalizable columns (recommended)

DROP_COLS = [
    "No.",
    "Info",
    "source_file",
    "Frame Time",
    "Frame Time (Epoch)",
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")


# 1) Prepare X/y

y_raw = df["label"].astype(str)
X_all = df.drop(columns=["label"])

# Drop all-NaN columns
all_nan_cols = [c for c in X_all.columns if X_all[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN columns:", all_nan_cols)
    X_all = X_all.drop(columns=all_nan_cols)

# Encode labels
le = LabelEncoder()
y_all = le.fit_transform(y_raw)

# Make y a Series aligned with X index (safer for iloc)
y_all_series = pd.Series(y_all, index=X_all.index)


# 2) Group-based split (prevents flow/IP overlap between train/test)

if "TCP Stream" in df.columns:
    groups = df["TCP Stream"].fillna(-1).astype(int)
    print("[INFO] Using GroupShuffleSplit groups = TCP Stream")
else:
    has_ip = ("IP Source" in df.columns) and ("IP Destination" in df.columns)
    if has_ip and ("TCP Source Port" in df.columns) and ("TCP Destination Port" in df.columns):
        groups = (
            df["IP Source"].astype(str)
            + "->"
            + df["IP Destination"].astype(str)
            + ":"
            + df["TCP Source Port"].astype(str)
            + "->"
            + df["TCP Destination Port"].astype(str)
        )
        print("[INFO] Using GroupShuffleSplit groups = IP pair + ports")
    elif has_ip:
        groups = df["IP Source"].astype(str) + "->" + df["IP Destination"].astype(str)
        print("[INFO] Using GroupShuffleSplit groups = IP pair")
    else:
        raise ValueError(
            "No suitable grouping columns found for group split. "
            "Need at least 'TCP Stream' OR ('IP Source' and 'IP Destination')."
        )

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_all, y_all_series, groups=groups))

# Pylance-friendly + safe indexing
train_idx = np.asarray(train_idx, dtype=np.int64)
test_idx = np.asarray(test_idx, dtype=np.int64)

X_train = X_all.iloc[train_idx].copy()
X_test = X_all.iloc[test_idx].copy()

y_train = y_all_series.iloc[train_idx].to_numpy(dtype=np.int64)
y_test = y_all_series.iloc[test_idx].to_numpy(dtype=np.int64)

y_raw_train = y_raw.iloc[train_idx].copy()
y_raw_test = y_raw.iloc[test_idx].copy()

print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)


# PART A) Classic ML (4 models)

numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

print("\nClassic ML column types")
print("Numeric:", len(numeric_cols))
print("Categorical:", len(categorical_cols))
if len(categorical_cols) > 0:
    print("Example categorical cols:", categorical_cols[:10])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

preprocess_classic = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop"
)

classic_models = {
    "random_forest": Pipeline(steps=[
        ("preprocess", preprocess_classic),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),
    "logistic_regression": Pipeline(steps=[
        ("preprocess", preprocess_classic),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "linear_svc": Pipeline(steps=[
        ("preprocess", preprocess_classic),
        ("clf", LinearSVC())
    ]),
    "sgd_classifier": Pipeline(steps=[
        ("preprocess", preprocess_classic),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE
        ))
    ]),
}

for name, model in classic_models.items():
    print("\n" + "=" * 70)
    print(f"Training CLASSIC model: {name}")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    f1w = f1_score(y_test, pred, average="weighted")
    print("Weighted F1:", f1w)
    print(classification_report(y_test, pred))

    out_path = RESULTS_DIR / f"model_{name}.joblib"
    joblib.dump(model, out_path)
    print("Saved:", out_path)

print("\n Classic models saved to results/model_*.joblib")


# PART B) Deep Learning (2 models): LSTM + 1D-CNN
# IMPORTANT: use ONLY numeric features to avoid OneHot explosion

num_cols_dl = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
if len(num_cols_dl) == 0:
    raise ValueError("No numeric columns found for deep learning models.")

Xtr_num = X_train[num_cols_dl].copy()
Xte_num = X_test[num_cols_dl].copy()

# Impute + standardize (save these to reuse during scoring)
dl_imputer = SimpleImputer(strategy="median")
Xtr_num = dl_imputer.fit_transform(Xtr_num)
Xte_num = dl_imputer.transform(Xte_num)

dl_mean = Xtr_num.mean(axis=0)
dl_std = Xtr_num.std(axis=0)
dl_std[dl_std == 0] = 1.0

Xtr_num = (Xtr_num - dl_mean) / dl_std
Xte_num = (Xte_num - dl_mean) / dl_std

n_features = Xtr_num.shape[1]
n_classes = int(len(np.unique(y_train)))

# Save DL preprocessing
joblib.dump(
    {
        "num_cols_dl": num_cols_dl,
        "imputer": dl_imputer,
        "mean": dl_mean,
        "std": dl_std,
        "label_classes": le.classes_.tolist(),
        "random_state": RANDOM_STATE,
    },
    DL_PREPROC_PATH
)
print("\nSaved DL preprocessing to:", DL_PREPROC_PATH)

# For LSTM/CNN treat features as a sequence: (samples, timesteps=n_features, channels=1)
Xtr_seq = Xtr_num.reshape((Xtr_num.shape[0], n_features, 1))
Xte_seq = Xte_num.reshape((Xte_num.shape[0], n_features, 1))

print("DL input shape:", Xtr_seq.shape, "(train)")

# Proper validation split for DL (stratified) instead of Keras validation_split
Xtr_seq_train, Xtr_seq_val, y_train_train, y_train_val = train_test_split(
    Xtr_seq, y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)


# DL1) LSTM
print("\n" + "=" * 70)
print("Training DEEP model: LSTM")

lstm = models.Sequential([
    layers.Input(shape=(n_features, 1)),
    layers.LSTM(64),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation="softmax"),
])

lstm.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

lstm.fit(
    Xtr_seq_train, y_train_train,
    validation_data=(Xtr_seq_val, y_train_val),
    epochs=5,
    batch_size=512,
    verbose="auto"
)

lstm.save(LSTM_PATH)
print("Saved LSTM to:", LSTM_PATH)


# DL2) 1D-CNN (Conv1D)
print("\n" + "=" * 70)
print("Training DEEP model: 1D-CNN")

cnn = models.Sequential([
    layers.Input(shape=(n_features, 1)),
    layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation="softmax"),
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn.fit(
    Xtr_seq_train, y_train_train,
    validation_data=(Xtr_seq_val, y_train_val),
    epochs=5,
    batch_size=512,
    verbose="auto"
)

cnn.save(CNN_PATH)
print("Saved CNN to:", CNN_PATH)

print("\n All models trained & saved.")
