import csv
import sys
from pathlib import Path
import re
from collections import defaultdict

COLS = ["accuracy@1", "accuracy@5", "accuracy@10"]

def natural_key(bug_id: str):
    m = re.match(r"^(.*?)-(\d+)$", bug_id)
    if m:
        return (m.group(1), int(m.group(2)))
    return (bug_id, float("inf"))

def parse_file(csv_path: Path) -> dict[str, set[str]]:
    """Return {col: set(bugs)} for expected COLS, ignoring blanks."""
    per_col = {c: set() for c in COLS}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # normalize headers (some CSVs might have slight variations in case/whitespace)
        field_map = {h.strip(): h for h in reader.fieldnames or []}
        for row in reader:
            for c in COLS:
                src = field_map.get(c, c)
                cell = (row.get(src, "") or "").strip()
                if cell:
                    per_col[c].add(cell)
    return per_col

def combine_sets(files_data: list[dict[str, set[str]]]):
    """Compute per-column union & intersection across all input files."""
    union = {c: set() for c in COLS}
    inter = {c: set() for c in COLS}
    for i, data in enumerate(files_data):
        for c in COLS:
            if i == 0:
                inter[c] = set(data[c])  # seed intersection
            else:
                inter[c] &= data[c]
            union[c] |= data[c]
    return union, inter

def write_wide(csv_path: Path, columns: dict[str, set[str]]):
    """Write a CSV with 3 columns (accuracy@1/@5/@10). Rows are padded."""
    sorted_cols = {c: sorted(columns[c], key=natural_key) for c in COLS}
    max_len = max(len(v) for v in sorted_cols.values()) if sorted_cols else 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i in range(max_len):
            row = []
            for c in COLS:
                row.append(sorted_cols[c][i] if i < len(sorted_cols[c]) else "")
            w.writerow(row)

def main(f1: str, f2: str, f3: str,
         out_union="bugs_union_per_k.csv",
         out_intersection="bugs_intersection_per_k.csv"):
    files_data = [parse_file(Path(p)) for p in (f1, f2, f3)]
    union, inter = combine_sets(files_data)
    write_wide(Path(out_union), union)
    write_wide(Path(out_intersection), inter)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script_name.py trial1.csv trial2.csv trial3.csv [union_out] [intersection_out]")
        sys.exit(1)
    main(*sys.argv[1:4], *sys.argv[4:])