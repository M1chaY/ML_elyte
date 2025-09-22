import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from rdkit import Chem


SMILES_COL_CANDIDATES = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "smiles_str",
    "smile",
]


def detect_smiles_column(columns: Iterable[str]) -> Optional[str]:
    """Detect the SMILES column name from a list of columns.

    Args:
        columns: Iterable of column names.

    Returns:
        The detected SMILES column name or None if not found.
    """
    lowered = {c.lower(): c for c in columns}
    for cand in SMILES_COL_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    # fallback: any column literally named like smiles-ish
    for c in columns:
        if "smile" in c.lower():
            return c
    return None


def validate_smiles(s: str) -> Tuple[bool, Optional[str]]:
    """Validate a SMILES string with RDKit.

    Args:
        s: SMILES string.

    Returns:
        (is_valid, error_message). error_message is None when valid.
    """
    if not isinstance(s, str) or not s.strip():
        return False, "empty-or-nonstring"
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return False, "MolFromSmiles returned None"
        # Round-trip to canonical to ensure sanitization
        _ = Chem.MolToSmiles(mol)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def iter_csv_chunks(path: Path, smiles_col: str, chunksize: int = 20000):
    """Yield (global_row_index, smiles_value) pairs from a CSV in chunks.

    Args:
        path: CSV file path.
        smiles_col: Column name containing SMILES.
        chunksize: Number of rows per pandas chunk.

    Yields:
        Tuples of (row_index, smiles_value).
    """
    row_offset = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if smiles_col not in chunk.columns:
            # In case of mixed header rows or weird CSV, skip this chunk
            row_offset += len(chunk)
            continue
        smiles_series = chunk[smiles_col]
        for i, v in smiles_series.items():
            yield i, v
        row_offset += len(chunk)


def scan_csv(path: Path) -> Tuple[int, int, List[Tuple[int, str, str]], Optional[str]]:
    """Scan a CSV file for SMILES validity.

    Args:
        path: CSV file path.

    Returns:
        total_rows, valid_count, invalid_rows(list of (row, smiles, error)), smiles_col.
    """
    # Try to detect column by peeking the header
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception:
        # fallback with pandas
        try:
            df_head = pd.read_csv(path, nrows=0)
            header = list(df_head.columns)
        except Exception:
            return 0, 0, [(0, "", "failed-to-read-csv")], None

    smiles_col = detect_smiles_column(header)
    if not smiles_col:
        return 0, 0, [(0, "", "smiles-column-not-found")], None

    total = 0
    valid = 0
    invalid: List[Tuple[int, str, str]] = []

    for row_idx, s in iter_csv_chunks(path, smiles_col):
        total += 1
        ok, err = validate_smiles(s)
        if ok:
            valid += 1
        else:
            invalid.append((row_idx, s if isinstance(s, str) else str(s), err or "invalid"))
        if total % 50000 == 0:
            print(f"  Scanned {total} rows... valid={valid} invalid={len(invalid)}")

    return total, valid, invalid, smiles_col


def write_invalid_report(path: Path, invalid_rows: List[Tuple[int, str, str]]) -> Optional[Path]:
    """Write invalid rows report next to the CSV.

    Args:
        path: Original CSV path.
        invalid_rows: List of (row_index, smiles, error).

    Returns:
        Path to the report if written, else None.
    """
    if not invalid_rows:
        return None
    report = path.with_suffix(path.suffix + ".invalid_smiles.csv")
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "smiles", "error"])  # header
        w.writerows(invalid_rows)
    return report


def find_csv_targets(base: Path) -> List[Path]:
    """Find CSV files to scan.

    Args:
        base: File or directory.

    Returns:
        List of CSV file paths.
    """
    if base.is_file() and base.suffix.lower() == ".csv":
        return [base]
    if base.is_dir():
        return sorted([p for p in base.rglob("*.csv")])
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SMILES in CSV files.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(Path.cwd()),
        help="CSV file or directory to scan (default: current workspace)",
    )
    args = parser.parse_args()

    targets = find_csv_targets(Path(args.path))
    if not targets:
        print("No CSV files found.")
        return 1

    print(f"Found {len(targets)} CSV file(s) to scan.")
    failures = 0

    for csv_path in targets:
        print(f"\nChecking: {csv_path}")
        total, valid, invalid_rows, smiles_col = scan_csv(csv_path)
        if smiles_col is None:
            print("  Skipped: SMILES column not found")
            continue
        print(f"  Column: {smiles_col}")
        print(f"  Total: {total}")
        print(f"  Valid: {valid}")
        print(f"  Invalid: {len(invalid_rows)}")
        report = write_invalid_report(csv_path, invalid_rows)
        if report:
            print(f"  Invalid rows saved to: {report}")
        if invalid_rows:
            failures += 1

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

