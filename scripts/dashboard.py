import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================
# CONFIG
# ==========================
DATASET_PATH = r"C:\Users\saeed khalilian\Desktop\insider"
MERGED_FILE = os.path.join(DATASET_PATH, "merged_attacks.csv")


# ---------- helpers ----------

def find_column(df, candidates):
    """
    Try to find a column whose name contains one of the candidate strings.
    Returns the real column name or None.
    """
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        cand_lower = cand.lower()
        for real_col, col_lower in zip(df.columns, cols):
            if cand_lower in col_lower:
                return real_col
    return None


# ---------- load data ----------

def load_data():
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(
            f"{MERGED_FILE} not found. Run attack_detection.py first to create it."
        )
    print(f"[INFO] Loading {MERGED_FILE}")
    df = pd.read_csv(MERGED_FILE, encoding="latin1", engine="python", on_bad_lines="skip")
    print("[INFO] Shape:", df.shape)
    print("[INFO] Columns (first 15):", list(df.columns)[:15])
    return df


# ---------- ML model to get feature importance ----------

def build_features(df):
    if "label" not in df.columns:
        raise RuntimeError("Column 'label' missing.")

    y = df["label"]

    # numeric features
    X = df.select_dtypes(include=["int64", "float64"]).copy()

    # add a couple of simple HTTP-based features if available
    if "HTTP Full URI" in df.columns:
        X["uri_length"] = df["HTTP Full URI"].fillna("").astype(str).str.len()
    if "HTTP Request URI" in df.columns:
        tmp = df["HTTP Request URI"].fillna("").astype(str)
        X["uri_special_chars"] = tmp.str.count(r"[?&=;'\"<>]")

    X = X.fillna(0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le


def train_model_for_importance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------- dashboard plots ----------

def create_dashboard(df, clf, label_encoder):
    # Prepare figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # ---- Panel 1: Attack distribution ----
    label_counts = df["label"].value_counts()
    ax1.bar(label_counts.index, label_counts.values)
    ax1.set_title("Attack Type Distribution")
    ax1.set_xlabel("Attack type")
    ax1.set_ylabel("Number of flows")

    # ---- Panel 2: Top source IPs (if column exists) ----
    src_col = find_column(df, ["ip source", "source ip", "src ip", "source"])
    if src_col:
        top_ips = df[src_col].value_counts().head(10)
        ax2.barh(top_ips.index.astype(str), top_ips.values)
        ax2.set_title(f"Top 10 Source IPs ({src_col})")
        ax2.set_xlabel("Number of flows")
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "No Source IP column found", ha="center", va="center")
        ax2.set_axis_off()

    # ---- Panel 3: Feature importance from RandomForest ----
    feature_importances = clf.feature_importances_
    feature_names = list(df.select_dtypes(include=["int64", "float64"]).columns)
    # Add the extra engineered features if present
    if "HTTP Full URI" in df.columns:
        feature_names = feature_names + ["uri_length"]
    if "HTTP Request URI" in df.columns:
        feature_names = feature_names + ["uri_special_chars"]

    # Sort top 15 important features
    feat_imp = sorted(
        zip(feature_names, feature_importances),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    names = [f[0] for f in feat_imp]
    values = [f[1] for f in feat_imp]

    ax3.barh(names, values)
    ax3.set_title("Top 15 Feature Importances")
    ax3.set_xlabel("Importance")
    ax3.invert_yaxis()

    # ---- Panel 4: Per-class counts over a numeric feature (e.g., packet length) ----
    # Try to find some "length"-like column
    length_col = find_column(df, ["length", "packet length", "ip length"])
    if length_col:
        for label in df["label"].unique():
            subset = df[df["label"] == label][length_col]
            ax4.hist(subset, bins=50, alpha=0.5, label=label)
        ax4.set_title(f"{length_col} distribution per attack type")
        ax4.set_xlabel(length_col)
        ax4.set_ylabel("Frequency")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No length-related column found", ha="center", va="center")
        ax4.set_axis_off()

    plt.suptitle("Network Attack Detection Dashboard", fontsize=16)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.subplots_adjust(top=0.92)
    plt.show()


def main():
    df = load_data()
    X, y, le = build_features(df)
    clf = train_model_for_importance(X, y)
    create_dashboard(df, clf, le)


if __name__ == "__main__":
    main()
