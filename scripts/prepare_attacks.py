import pandas as pd
from config import DATA_RAW, DATA_PROCESSED

# File → Label mapping
FILES = {
    "DDOS-UDP.csv": "ddos_udp",
    "DOS_csvvw.csv": "dos",
    "Remote_Code_Execution.csv": "rce",
    "SQL Injection.csv": "sqli",
    "SQL-Injection2.csv": "sqli",
    "SQLInjection-UPDATED.csv": "sqli",
}

def load_and_label() -> pd.DataFrame:
    dfs = []

    for filename, label in FILES.items():
        path = DATA_RAW / filename

        if not path.exists():
            print(f"[WARNING] File not found: {path}")
            continue

        print(f"[INFO] Loading {filename} → label={label}")

        df = pd.read_csv(
            path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        df["label"] = label
        df["source_file"] = filename

        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            f"No CSV files were loaded. "
            f"Please check files in {DATA_RAW}"
        )

    merged = pd.concat(dfs, ignore_index=True)

    print("\n[INFO] Combined dataset shape:", merged.shape)
    print("[INFO] Columns found (first 20):", list(merged.columns)[:20], "...")

    return merged

if __name__ == "__main__":
    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    merged_df = load_and_label()

    output_path = DATA_PROCESSED / "merged_attacks.csv"
    merged_df.to_csv(output_path, index=False)

    print(f"\n[OK] Saved merged dataset to: {output_path}")
