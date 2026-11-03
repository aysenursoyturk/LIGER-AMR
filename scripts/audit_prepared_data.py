#!/usr/bin/env python3
"""Read-only audit of the copied final nested-CV data; no third-party packages."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ANTIBIOTICS = ("CIP", "CLI", "Ery", "FUS", "GEN", "MET", "TET", "TMP")
REQUIRED = ("train_snp.npz", "test_snp.npz", "train_gene.npz", "test_gene.npz", "y_train.npy", "y_test.npy", "fold_config.json")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    errors = []
    checked = 0
    matched = 0
    for antibiotic in ANTIBIOTICS:
        for fold in range(1, 6):
            rel = Path(antibiotic) / f"outer_fold_{fold}"
            target = args.data_root / rel
            for filename in REQUIRED:
                if not (target / filename).is_file():
                    errors.append(f"MISSING: {target / filename}")
            config_file = target / "fold_config.json"
            if config_file.is_file():
                try:
                    config = json.loads(config_file.read_text())
                    if config.get("antibiotic") != antibiotic or int(config.get("outer_fold", -1)) != fold:
                        errors.append(f"CONFIG MISMATCH: {config_file}")
                    if config.get("train_test_overlap") != 0:
                        errors.append(f"OVERLAP FLAG: {config_file}")
                    if config.get("outer_test_used_for_feature_selection") is not False:
                        errors.append(f"FEATURE SELECTION FLAG: {config_file}")
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"INVALID CONFIG: {config_file}: {exc}")
            if args.source_root is not None:
                original = args.source_root / rel
                if not original.is_dir():
                    errors.append(f"SOURCE FOLDER MISSING: {original}")
                    continue
                source_names = {p.name for p in original.iterdir() if p.is_file()}
                target_names = {p.name for p in target.iterdir() if p.is_file()} if target.is_dir() else set()
                if source_names != target_names:
                    errors.append(f"FILE SET DIFF: {rel} source-only={sorted(source_names-target_names)} target-only={sorted(target_names-source_names)}")
                for filename in sorted(source_names & target_names):
                    checked += 1
                    if sha256(original / filename) != sha256(target / filename):
                        errors.append(f"SHA256 MISMATCH: {rel / filename}")
                    else:
                        matched += 1
    print(f"Expected folds: {len(ANTIBIOTICS)*5}")
    print(f"SHA256 comparisons: {checked}; matched: {matched}")
    print(f"Issues: {len(errors)}")
    for issue in errors[:50]:
        print(issue)
    if len(errors) > 50:
        print(f"... {len(errors)-50} more issues")
    raise SystemExit(1 if errors else 0)

if __name__ == "__main__":
    main()
