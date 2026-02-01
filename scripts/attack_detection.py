import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import DATA_RAW, DATA_PROCESSED, RESULTS_DIR


# Map each CSV file to its attack label (from data/raw/)
FILE_LABELS = {
    "DDOS-UDP.csv": "ddos_udp",
    "DOS_csvvw.csv": "dos",  # optional (remove if you don't have it)
    "Remote_Code_Execution.csv": "rce",
    "SQL Injection.csv": "sqli",
    "SQL-Injection2.csv": "sqli",
    "SQLInjection-UPDATED.csv": "sqli",
}

MERGED_FILE = DATA_PROCESSED / "merged_attacks.csv"


def merge_csv_files() -> pd.DataFrame:
    """
    Read all attack CSV files from data/raw/, add a 'label' column,
    and save a single data/processed/merged_attacks.csv file.
    """
    dfs = []

    for filename, label in FILE_LABELS.items():
        path = DATA_RAW / filename

        if not path.exists():
            print(f"[WARNING] File not found, skipping: {path}")
            continue

        print(f"[INFO] Loading {filename} -> label={label}")
        df = pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip")

        df["label"] = label
        df["source_file"] = filename
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files were loaded. Put raw CSVs into {DATA_RAW} "
            f"and verify FILE_LABELS names."
        )

    merged = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Merged dataset shape: {merged.shape}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_FILE, index=False)
    print(f"[OK] Saved merged file to: {MERGED_FILE}")

    return merged


def load_merged() -> pd.DataFrame:
    """
    Load data/processed/merged_attacks.csv if it exists, otherwise create it by merging.
    """
    if MERGED_FILE.exists():
        print(f"[INFO] Loading existing merged file: {MERGED_FILE}")
        df = pd.read_csv(MERGED_FILE, encoding="latin1", engine="python", on_bad_lines="skip")
        print(f"[INFO] Merged dataset shape: {df.shape}")
        return df

    print("[INFO] merged_attacks.csv not found, creating it now from data/raw/ ...")
    return merge_csv_files()


def make_features(df: pd.DataFrame):
    """
    Build the feature matrix X and target vector y.

    Features:
      - all numeric columns
      - + 2 simple features based on HTTP URIs (if they exist).
    """
    if "label" not in df.columns:
        raise RuntimeError("Column 'label' missing from DataFrame.")

    y = df["label"].astype(str)

    X = df.select_dtypes(include=["int64", "float64"]).copy()

    if "HTTP Full URI" in df.columns:
        X["uri_length"] = df["HTTP Full URI"].fillna("").astype(str).str.len()

    if "HTTP Request URI" in df.columns:
        tmp = df["HTTP Request URI"].fillna("").astype(str)
        X["uri_special_chars"] = tmp.str.count(r"[?&=;'\"<>]")

    X = X.fillna(0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"[INFO] Number of feature columns: {X.shape[1]}")
    print("[INFO] Classes:", list(le.classes_))

    return X, y_enc, le


def train_and_evaluate(X, y, label_encoder: LabelEncoder):
    """
    Split into train/test, train RandomForest, and show metrics & confusion matrix.
    Also saves confusion matrix image into results/.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("[INFO] Train shape:", X_train.shape, " Test shape:", X_test.shape)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    print("[INFO] Training RandomForest...")
    clf.fit(X_train, y_train)

    print("[INFO] Predicting on test set...")
    y_pred = clf.predict(X_test)

    class_names = list(label_encoder.classes_)

    print("\n========== Classification Report ==========")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("========== Confusion Matrix ==========")
    print(cm)

    # Plot + save confusion matrix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()

    out_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] Confusion matrix image: {out_path}")

    plt.show()

    return clf


def main():
    df = load_merged()
    X, y, le = make_features(df)
    _ = train_and_evaluate(X, y, le)


if __name__ == "__main__":
    main()
