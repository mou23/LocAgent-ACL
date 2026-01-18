#!/usr/bin/env python3
import os
import json
import argparse
from glob import glob
import re
import csv
from tqdm import tqdm
from datasets import load_dataset

def check_localization_at_k(fixed_files, suspicious_files, k):
    """Return True if any fixed file is in the top-k suspicious files."""
    top_files = suspicious_files[:k]
    for fixed_file in fixed_files:
        if fixed_file in top_files:
            return True
    return False

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {ln} in {path}: {e}")


def extract_fixed_files(patch: str):
    files = set()
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            # diff --git a/foo/bar.py b/foo/bar.py
            file_path = parts[2][2:]  # remove "a/"
            files.add(file_path)
    return sorted(files)


def parse_files_from_raw_output_loc(raw_output_loc):
    """
    raw_output_loc: list[str] (often length 1), containing narrative text with a ```...``` block.
    Returns: list[str] of file paths (e.g., 'astropy/io/ascii/rst.py')
    """
    if not raw_output_loc:
        return []

    text = "\n".join([s for s in raw_output_loc if isinstance(s, str)])

    # Extract all fenced code blocks ``` ... ```
    blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return []

    files = []
    for blk in blocks:
        for line in blk.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Find a path ending in .py at the start of the line
            # Examples:
            #   astropy/io/ascii/rst.py:RST
            #   astropy/io/ascii/fixedwidth.py:FixedWidthData.write
            m = re.match(r"^(.+?\.py)\b", line)
            if m:
                files.append(m.group(1))

    # de-dup while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def load_output(root,trial):
    pattern = os.path.join(root, f"swe-res-{trial}/location", "loc_outputs.jsonl")
    paths = glob(pattern)
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")

    results = {}
    for p in paths:
        project = os.path.basename(os.path.dirname(p))
        print(f"Reading jsonl: {project} -> {p}")

        for rec in read_jsonl(p):
            instance_id = str(rec.get("instance_id", "")).strip()

            raw_found = rec.get("found_files") or []
            raw_output_loc = rec.get("raw_output_loc") or []

            # Normalize found_files to a flat list
            found_files = []
            if isinstance(raw_found, list):
                for item in raw_found:
                    if isinstance(item, list):
                        found_files.extend(item)
                    elif isinstance(item, str):
                        found_files.append(item)
            elif isinstance(raw_found, str):
                found_files = [raw_found]

            # If empty (covers [] and [[]]), fall back to parsing raw_output_loc
            if not found_files:
                found_files = parse_files_from_raw_output_loc(raw_output_loc)
                # print(instance_id,found_files)
            results[instance_id] = found_files

    return results


def main(root):
    trial = 3
    results = load_output(root,trial)

    # SWE-bench Lite test split
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    bug_results = {}
    missing_result = 0
    missing_fixed = 0

    for issue in dataset:
        instance_id = str(issue.get("instance_id", "")).strip()
        if not instance_id:
            continue

        patch = issue.get("patch") 
        fixed_files = extract_fixed_files(patch)


        if not fixed_files:
            missing_fixed += 1
            # keep it out of evaluation, like your "skip if fixed missing"
            continue

        suspicious_files = results.get(instance_id, [])
        if not suspicious_files:
            missing_result += 1

        bug_results[instance_id] = {
            "suspicious_files": suspicious_files,
            "fixed_files": fixed_files,
        }
    acc1_ids, acc5_ids, acc10_ids = [], [], []

    for bug_id, bug in tqdm(bug_results.items(), desc="Processing Bugs"):
        fixed_files = bug.get('fixed_files', []) or []
        suspicious_files = bug.get('suspicious_files', []) or []

        if check_localization_at_k(fixed_files, suspicious_files, 1):
            acc1_ids.append(bug_id)
        if check_localization_at_k(fixed_files, suspicious_files, 5):
            acc5_ids.append(bug_id)
        if check_localization_at_k(fixed_files, suspicious_files, 10):
            acc10_ids.append(bug_id)

    # Align rows so CSV has equal length columns (fill empty cells with "")
    max_len = max(len(acc1_ids), len(acc5_ids), len(acc10_ids))
    rows = []
    for i in range(max_len):
        rows.append([
            acc1_ids[i] if i < len(acc1_ids) else "",
            acc5_ids[i] if i < len(acc5_ids) else "",
            acc10_ids[i] if i < len(acc10_ids) else ""
        ])

    out_file = f"localized_bugs{trial}.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy@1", "accuracy@5", "accuracy@10"])
        writer.writerows(rows)

    print(f"Wrote bug IDs to {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate combined_locs.jsonl on SWE-bench Lite using patch-derived fixed files.")
    ap.add_argument("--root", default=".", help="Repo root (default: current dir)")
    args = ap.parse_args()
    main(args.root)