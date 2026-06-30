from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, type=Path,
                   help="run directories, each containing predictions.npz")
    p.add_argument("--out", type=Path, default=None,
                   help="optional path to save ensemble predictions npz")
    return p.parse_args()


def main():
    args = parse_args()

    test_preds_list = []
    val_maes = []
    test_truths_ref = None
    target_names = None

    print("loading individual runs:")
    for run_dir in args.runs:
        d = np.load(run_dir / "predictions.npz", allow_pickle=True)
        test_preds_list.append(d["test_preds"])
        if test_truths_ref is None:
            test_truths_ref = d["test_truths"]
            target_names = d["target_names"]
        else:
            if not np.allclose(test_truths_ref, d["test_truths"]):
                raise ValueError(
                    f"test truths in {run_dir} differ from the first run; "
                    "ensemble averaging requires a fixed test set "
                    "(use the same --test-seed across runs)"
                )
        val_mae = float(np.abs(d["val_preds"] - d["val_truths"]).mean())
        val_maes.append(val_mae)
        print(f"  {run_dir.name:>20s}  val_MAE={val_mae:.4f}")

    val_maes = np.array(val_maes)
    inv = 1.0 / val_maes
    weights = inv / inv.sum()
    print(f"\nensemble weights: {weights.round(4).tolist()}")

    test_preds = np.stack(test_preds_list)
    ensemble_preds = (test_preds * weights[:, None, None]).sum(axis=0)

    mae = np.abs(ensemble_preds - test_truths_ref).mean(axis=0)
    rmse = np.sqrt(((ensemble_preds - test_truths_ref) ** 2).mean(axis=0))

    print("\nensemble test set per-target metrics:")
    for name, m, r in zip(target_names, mae, rmse):
        print(f"  {str(name):>8s}  MAE={m:.4f}  RMSE={r:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out,
                 ensemble_preds=ensemble_preds,
                 test_truths=test_truths_ref,
                 weights=weights,
                 val_maes=val_maes,
                 target_names=target_names)
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()