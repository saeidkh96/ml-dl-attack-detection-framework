import pandas as pd
from config import DATA_RAW, DATA_PROCESSED

FILE_LABELS = {
    "DDOS-UDP.csv": "ddos_udp",
    "DOS_csvvw.csv": "dos",
    "Remote_Code_Execution.csv": "rce",
    "SQL Injection.csv": "sqli",
    "SQL-Injection2.csv": "sqli",
    "SQLInjection-UPDATED.csv": "sqli",
}

def load_and_merge() -> pd.DataFrame:
    dfs = []

    for filename, label in FILE_LABELS.items():
        path = DATA_RAW / filename

        if not path.exists():
            print(f"[WARNING] File missing: {filename}  (expected at: {path})")
            continue

        print(f"[INFO] Loading: {filename}  → label={label}")

        df = pd.read_csv(
            path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        # Add attack label
        df["label"] = label

        # Add source filename for traceability
        df["source_file"] = filename

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No input CSV files were found in {DATA_RAW}. "
            "Please place your raw CSV files into data/raw/."
        )

    merged = pd.concat(dfs, ignore_index=True)

    print("\n[INFO] Final merged shape:", merged.shape)
    print("[INFO] Columns (first 15):", list(merged.columns)[:15], "...")

    return merged

if __name__ == "__main__":
    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    merged_df = load_and_merge()

    output_path = DATA_PROCESSED / "merged_attacks.csv"
    merged_df.to_csv(output_path, index=False)

    print(f"\n[SAVED] merged_attacks.csv created at: {output_path}")
