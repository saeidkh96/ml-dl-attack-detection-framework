# scripts/train_model.py

from __future__ import annotations

import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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


# Load data

df = pd.read_csv(MERGED_FILE, low_memory=False)
if "label" not in df.columns:
    raise ValueError("❌ Column 'label' not found in dataset")

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

# Fixed split for fair comparisons
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X_all, y_all, y_raw,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_all
)

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
    Xtr_seq, y_train,
    validation_split=0.2,
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
    Xtr_seq, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=512,
    verbose="auto"
)

cnn.save(CNN_PATH)
print("Saved CNN to:", CNN_PATH)

print("\n All models trained & saved.")
