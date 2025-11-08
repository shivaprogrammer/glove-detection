#!/usr/bin/env python3
"""
remap_labels.py

Usage:
  python remap_labels.py --labels_root /data/datasets/data/glove_detection/train/labels \
                         --mapping 0:0,1:1,2:0

Meaning of mapping:
  old_class:new_class,old_class2:new_class2,...
Example mapping to merge classes 0 and 2 into class 0, keep 1 -> 1:
  0:0,1:1,2:0
"""
import argparse
from pathlib import Path

def parse_mapping(s):
    m = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        a,b = pair.split(":")
        m[int(a)] = int(b)
    return m

def remap_folder(labels_root: Path, mapping: dict, dry_run=False):
    labels = list(labels_root.glob("*.txt"))
    print(f"Found {len(labels)} label files in {labels_root}")
    for f in labels:
        txt = f.read_text().strip().splitlines()
        if not txt:
            continue
        new_lines = []
        changed = False
        for line in txt:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(float(parts[0]))
            if old_cls not in mapping:
                raise ValueError(f"Old class {old_cls} not in mapping")
            new_cls = mapping[old_cls]
            if new_cls != old_cls:
                changed = True
            new_line = " ".join([str(new_cls)] + parts[1:])
            new_lines.append(new_line)
        if changed and not dry_run:
            f.write_text("\n".join(new_lines) + "\n")
    print("Remapping done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_root", required=True, help="folder containing label .txt files")
    p.add_argument("--mapping", required=True, help="map like 0:0,1:1,2:0")
    p.add_argument("--dry", action="store_true", help="dry run")
    args = p.parse_args()
    mapping = parse_mapping(args.mapping)
    remap_folder(Path(args.labels_root), mapping, dry_run=args.dry)
