from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fit_affine_calibration(preds: np.ndarray, truths: np.ndarray, target_idx: int = 0):
    x = preds[:, target_idx]
    y = truths[:, target_idx]
    design = np.stack([x, np.ones_like(x)], axis=1)
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def apply_target_affine(preds: np.ndarray, target_idx: int, slope: float, intercept: float):
    calibrated = preds.copy()
    calibrated[:, target_idx] = slope * calibrated[:, target_idx] + intercept
    return calibrated


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, type=Path,
                   help="run directories, each containing predictions.npz")
    p.add_argument("--out", type=Path, default=None,
                   help="optional path to save ensemble predictions npz")
    p.add_argument("--calibrate-hf-affine", action="store_true",
                   help="apply validation-set affine correction to Hf_298")
    return p.parse_args()


def main():
    args = parse_args()

    test_preds_list = []
    val_preds_list = []
    val_truths_list = []
    val_maes = []
    test_truths_ref = None
    target_names = None

    print("loading individual runs:")
    for run_dir in args.runs:
        d = np.load(run_dir / "predictions.npz", allow_pickle=True)
        test_preds_list.append(d["test_preds"])
        val_preds_list.append(d["val_preds"])
        val_truths_list.append(d["val_truths"])
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

    hf_calibration = np.array([1.0, 0.0], dtype=np.float32)
    if args.calibrate_hf_affine:
        calibration_val_preds = np.concatenate(val_preds_list, axis=0)
        calibration_val_truths = np.concatenate(val_truths_list, axis=0)
        slope, intercept = fit_affine_calibration(
            calibration_val_preds, calibration_val_truths, 0,
        )
        hf_calibration = np.array([slope, intercept], dtype=np.float32)
        ensemble_preds = apply_target_affine(ensemble_preds, 0, slope, intercept)
        print(f"Hf affine calibration from all val folds: y = {slope:.6f} * pred + {intercept:.6f}")

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
                 hf_calibration=hf_calibration,
                 target_names=target_names)
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()