#!/usr/bin/env python3
"""Validate a prepared COCO single dataset.

Trimmed to the COCO part of jeehoo0507/kshs-aimlab-benchmarks
scripts/check_assets.py (main @ e81cc30); the checks are unchanged.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


CLASSES = ("giraffe", "airplane", "clock", "zebra", "train", "bird",
           "elephant", "toilet", "stop sign", "bear")
SPLITS = {"train", "val", "test"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, name):
    rel = Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts and name, f"Unsafe relative path: {name}")
    root = root.resolve()
    path = (root / rel).resolve()
    require(path.is_relative_to(root), f"Path leaves root: {name}")
    require(path.is_file(), f"Missing file: {path}")
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_coco(root, full):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == 1, "Unsupported COCO manifest schema")
    require(manifest.get("selection") == "one_annotated_instance_per_image",
            "COCO manifest uses an older selection rule; move the old dataset aside and rerun setup")
    require(tuple(manifest.get("classes", [])) == CLASSES,
            "COCO class set differs; move the old dataset aside and rerun setup")
    require(manifest.get("split_seed") == 20260922, "COCO split seed differs from documented definition")
    rows = manifest.get("images")
    require(isinstance(rows, list) and rows, "COCO manifest has no images")
    counts, probes, ids, decoded = Counter(), Counter(), set(), set()
    for row in rows:
        sample_id = row["id"]
        require(row.get("instances") == 1, f"COCO image does not have exactly one annotated object: {sample_id}")
        split, label = row["split"], row["label"]
        require(split in SPLITS and type(label) is int and 0 <= label < len(CLASSES),
                f"Invalid split/label: {sample_id}")
        require(sample_id not in ids, f"Duplicate COCO image id: {sample_id}")
        ids.add(sample_id)
        source = row.get("source_split")
        require(source == ("val2017" if split == "test" else "train2017"),
                f"Source split mismatch: {sample_id}")
        if "pixel_sha256" in row:
            require(row["pixel_sha256"] not in decoded, f"Duplicate decoded image: {sample_id}")
            decoded.add(row["pixel_sha256"])
        counts[split, label] += 1
        if row.get("probe"):
            require(split == "val", f"Probe outside validation: {sample_id}")
            probes[label] += 1
        for field in ("image", "mask"):
            path = inside(root, row[field])
            expected = row.get(field + "_sha256")
            require(isinstance(expected, str) and SHA256.fullmatch(expected),
                    f"Missing/invalid {field} SHA-256: {sample_id}")
            if full:
                require(digest(path) == expected, f"Changed {field}: {sample_id}")
    for split in SPLITS:
        require(all(counts[split, label] > 0 for label in range(len(CLASSES))),
                f"Missing COCO class in {split}")
    for split in ("train", "val"):
        require(len({counts[split, label] for label in range(len(CLASSES))}) == 1,
                f"Unbalanced COCO {split} split")
    require(all(probes[label] == 20 for label in range(len(CLASSES))),
            "COCO probe should contain 20 validation images per class")
    print("COCO single verified" + (" (SHA-256 checked)" if full else " (file presence)"))
    print("  " + ", ".join(f"{split}={sum(counts[split, c] for c in range(10))}" for split in ("train", "val", "test")))
    print("  train/val per class:", counts["train", 0], counts["val", 0], "; probe=20 per class")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    coco = sub.add_parser("coco", help="check COCO single")
    coco.add_argument("root", type=Path)
    coco.add_argument("--full", action="store_true", help="hash every image and mask")
    args = parser.parse_args()
    try:
        check_coco(args.root, args.full)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Asset check failed: {exc}\n")


if __name__ == "__main__":
    main()
