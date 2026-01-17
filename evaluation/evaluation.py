#!/usr/bin/env python3
import os
import json
import argparse
from glob import glob
from datasets import load_dataset

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


def calculate_metrics(bug_results):
    total_bugs = len(bug_results)
    print(f"\nTotal Bugs Processed: {total_bugs}")
    if total_bugs == 0:
        return

    # Accuracy@k
    for top in [1, 5, 10]:
        count = 0
        for bug_id, result in bug_results.items():
            suspicious_files = result["suspicious_files"]
            fixed_files = result["fixed_files"]

            for fixed_file in fixed_files:
                if fixed_file in suspicious_files[:top]:
                    # print(bug_id, fixed_file)
                    count += 1
                    break
        print(f"Accuracy@{top}: {count}/{total_bugs} = {count*100/total_bugs:.2f}%")

    # MRR@10
    inverse_rank = 0
    for bug_id, result in bug_results.items():
        suspicious_files = result["suspicious_files"]
        fixed_files = result["fixed_files"]
        minimum_length = min(10, len(suspicious_files))
        for i in range(minimum_length):
            if suspicious_files[i] in fixed_files:
                inverse_rank += 1 / (i + 1)
                break
    mrr = inverse_rank / total_bugs
    print(f"MRR@10: {mrr:.4f}")

    # MAP@10
    total_average_precision = 0
    for bug_id, result in bug_results.items():
        suspicious_files = result["suspicious_files"]
        fixed_files = result["fixed_files"]

        if not fixed_files:
            continue

        precision_sum = 0
        relevant_so_far = 0
        minimum_length = min(10, len(suspicious_files))
        for i in range(minimum_length):
            if suspicious_files[i] in fixed_files:
                relevant_so_far += 1
                precision_sum += relevant_so_far / (i + 1)

        total_average_precision += (precision_sum / len(fixed_files))

    map_k = total_average_precision / total_bugs
    print(f"MAP@10: {map_k:.4f}")


import re

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


def load_output(root):
    pattern = os.path.join(root, "swe-res-1/location", "loc_outputs.jsonl")
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
                print(instance_id,found_files)
            results[instance_id] = found_files

    return results


def main(root):
    results = load_output(root)

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
    # print(bug_results)

    print(f"\nBuilt evaluation set: {len(bug_results)}")
    print(f"Skipped (no fixed_files from patch): {missing_fixed}")
    print(f"Instances with no found_files in jsonl: {missing_result}")

    calculate_metrics(bug_results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate combined_locs.jsonl on SWE-bench Lite using patch-derived fixed files.")
    ap.add_argument("--root", default=".", help="Repo root (default: current dir)")
    args = ap.parse_args()
    main(args.root)