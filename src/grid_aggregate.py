from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for d in args.runs:
        npz_path = d / "predictions.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        mae = np.abs(data["test_preds"] - data["test_truths"]).mean(axis=0)
        rows.append((d.name, mae))

    rows.sort(key=lambda r: r[1][0])

    targets = ["Hf", "S", "Cp300", "Cp400", "Cp500", "Cp600",
               "Cp800", "Cp1000", "Cp1500"]
    header = f"{'config':<25s}" + "".join(f"  {t:>7s}" for t in targets)
    print()
    print(header)
    print("-" * len(header))
    for name, mae in rows:
        line = f"{name:<25s}" + "".join(f"  {m:>7.3f}" for m in mae)
        print(line)


if __name__ == "__main__":
    main()