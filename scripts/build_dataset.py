# scripts/build_dataset.py

from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, MERGED_FILE


LABEL_CANDIDATES = ["label", "Label", "attack", "Attack", "class", "Class", "target", "Target", "y"]


def _find_label_column(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    # fuzzy match (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    for cand in [c.lower() for c in LABEL_CANDIDATES]:
        if cand in lower_map:
            return lower_map[cand]
    return None


def build_merged_dataset(
    raw_dir: Path | None = None,
    out_file: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Build a merged dataset CSV from any CSV files in data/raw/.

    Rules:
    - If out_file exists and force=False -> keep it (do nothing).
    - Read all *.csv in raw_dir.
    - If a CSV has a label-like column -> use it as 'label'.
    - Otherwise -> create 'label' from filename stem.
    - Save merged to out_file and return its path.
    """
    raw_dir = raw_dir or DATA_RAW
    out_file = out_file or MERGED_FILE

    if out_file.exists() and not force:
        print(f"[INFO] Using existing merged dataset: {out_file}")
        return out_file

    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            f"Put your dataset CSV(s) into {raw_dir} and try again."
        )

    dfs: list[pd.DataFrame] = []
    for fp in csv_files:
        print(f"[INFO] Loading: {fp.name}")
        df = pd.read_csv(fp, encoding="latin1", engine="python", on_bad_lines="skip")

        label_col = _find_label_column(df)
        if label_col is not None:
            # normalize label column name
            if label_col != "label":
                df = df.rename(columns={label_col: "label"})
        else:
            # no label column -> infer from filename
            df["label"] = fp.stem.lower()

        df["source_file"] = fp.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Merged shape: {merged.shape}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False)
    print(f"[OK] Saved merged dataset to: {out_file}")

    return out_file


if __name__ == "__main__":
    build_merged_dataset(force=False)
