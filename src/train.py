from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import build_dataloader
from chemprop.nn.transforms import UnscaleTransform

from dataset import (
    TARGET_COLS, load_datapoints, split_with_fixed_test, split_by_complexity,
    make_dataset,
)
from model import build_deeptherm


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


def atom_numbers(points):
    nums = set()
    for dp in points:
        nums.update(atom.GetAtomicNum() for atom in dp.mol.GetAtoms())
    return sorted(nums)


def atom_count_matrix(points, nums):
    pos = {num: i for i, num in enumerate(nums)}
    x = np.ones((len(points), len(nums) + 1), dtype=np.float32)
    x[:, 1:] = 0.0
    for row, dp in enumerate(points):
        for atom in dp.mol.GetAtoms():
            idx = pos.get(atom.GetAtomicNum())
            if idx is not None:
                x[row, idx + 1] += 1.0
    return x


def fit_hf_atomref(train_pts, all_pts):
    nums = atom_numbers(all_pts)
    x = atom_count_matrix(train_pts, nums)
    y = np.array([dp.y[0] for dp in train_pts], dtype=np.float32)
    coef = np.linalg.lstsq(x, y, rcond=None)[0].astype(np.float32)
    return nums, coef


def apply_hf_atomref(points, nums, coef):
    baseline = atom_count_matrix(points, nums) @ coef
    for dp, base in zip(points, baseline):
        dp.y[0] = dp.y[0] - base
    return baseline.astype(np.float32)


def add_hf_baseline(values: np.ndarray, baseline: np.ndarray):
    restored = values.copy()
    restored[:, 0] = restored[:, 0] + baseline
    return restored


def select_checkpoint_by_val_hf(trainer, model, val_loader, val_truths, checkpoint_dir):
    checkpoint_paths = sorted(Path(checkpoint_dir).glob("*.ckpt"))
    if not checkpoint_paths:
        return None
    best_path = None
    best_mae = float("inf")
    for path in checkpoint_paths:
        pred_batches = trainer.predict(model, val_loader, ckpt_path=str(path), weights_only=False)
        preds = torch.cat(pred_batches).cpu().numpy()
        mae = float(np.abs(preds[:, 0] - val_truths[:, 0]).mean())
        if mae < best_mae:
            best_mae = mae
            best_path = path
    print(f"best_hf_checkpoint: {best_path}  val_hf_mae={best_mae:.4f}")
    return str(best_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=Path("runs/default"))
    p.add_argument("--resume-ckpt", type=Path, default=None,
                   help="resume training from a checkpoint")
    p.add_argument("--init-from-ckpt", type=Path, default=None,
                   help="load model weights from a checkpoint")
    p.add_argument("--seed", type=int, default=42,
                   help="train/val split + model init seed")
    p.add_argument("--test-seed", type=int, default=42,
                   help="test set selection for random split mode")
    p.add_argument("--split-mode", choices=["random", "complexity"],
                   default="random",
                   help="random 81:9:10 or hierarchical complexity-based")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--d-hidden", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-hidden", type=int, default=300)
    p.add_argument("--ffn-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ecfp-bits", type=int, default=0,
                   help="Morgan fingerprint length; 0 disables ECFP descriptors")
    p.add_argument("--ecfp-proj-dim", type=int, default=64)
    p.add_argument("--ecfp-mode", choices=["projected", "direct"],
                   default="projected",
                   help="ECFP descriptor transform")
    p.add_argument("--ecfp-scale", type=float, default=0.01)
    p.add_argument("--trainable-ecfp-proj", action="store_true",
                   help="train the ECFP projection")
    p.add_argument("--calibrate-hf-affine", action="store_true",
                   help="apply validation-set affine correction to Hf_298")
    p.add_argument("--atomref-hf", action="store_true",
                   help="train Hf_298 as an atom-reference residual")
    p.add_argument("--hf-loss-weight", type=float, default=1.0,
                   help="loss weight for Hf_298")
    p.add_argument("--loss", choices=["mse", "mae"], default="mse")
    p.add_argument("--select-best-hf", action="store_true",
                   help="select checkpoint by validation Hf_298 MAE")
    p.add_argument("--init-lr", type=float, default=1e-4)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--final-lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)

    points = load_datapoints(args.csv, ecfp_bits=args.ecfp_bits)
    if args.split_mode == "complexity":
        train_pts, val_pts, test_pts = split_by_complexity(
            points, train_val_seed=args.seed,
        )
    else:
        train_pts, val_pts, test_pts = split_with_fixed_test(
            points, train_val_seed=args.seed, test_seed=args.test_seed,
        )
    print(f"split={args.split_mode}  "
          f"train={len(train_pts)}  val={len(val_pts)}  test={len(test_pts)}")

    atomref_nums = np.array([], dtype=np.int16)
    atomref_coef = np.array([], dtype=np.float32)
    train_hf_base = np.zeros(len(train_pts), dtype=np.float32)
    val_hf_base = np.zeros(len(val_pts), dtype=np.float32)
    test_hf_base = np.zeros(len(test_pts), dtype=np.float32)
    if args.atomref_hf:
        nums, coef = fit_hf_atomref(train_pts, points)
        atomref_nums = np.array(nums, dtype=np.int16)
        atomref_coef = coef
        train_hf_base = apply_hf_atomref(train_pts, nums, coef)
        val_hf_base = apply_hf_atomref(val_pts, nums, coef)
        test_hf_base = apply_hf_atomref(test_pts, nums, coef)
        print(f"atomref_hf={len(nums)} elements")

    train_ds = make_dataset(train_pts)
    val_ds = make_dataset(val_pts)
    test_ds = make_dataset(test_pts)

    target_scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(target_scaler)
    output_transform = UnscaleTransform.from_standard_scaler(target_scaler)

    train_loader = build_dataloader(train_ds, args.batch_size,
                                    args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_ds, args.batch_size,
                                  args.num_workers, shuffle=False)
    test_loader = build_dataloader(test_ds, args.batch_size,
                                   args.num_workers, shuffle=False)

    task_weights = torch.ones(len(TARGET_COLS), dtype=torch.float32)
    task_weights[0] = args.hf_loss_weight

    model = build_deeptherm(
        n_targets=len(TARGET_COLS),
        d_hidden=args.d_hidden,
        depth=args.depth,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        ffn_layers=args.ffn_layers,
        dropout=args.dropout,
        ecfp_bits=args.ecfp_bits,
        ecfp_proj_dim=args.ecfp_proj_dim,
        ecfp_mode=args.ecfp_mode,
        ecfp_scale=args.ecfp_scale,
        ecfp_trainable=args.trainable_ecfp_proj,
        task_weights=task_weights,
        loss=args.loss,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        output_transform=output_transform,
    )

    if args.init_from_ckpt is not None:
        ckpt = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        task_weights_device = task_weights.to(model.criterion.task_weights.device).view(1, -1)
        for metric in [model.criterion, *model.metrics]:
            if hasattr(metric, "task_weights") and metric.task_weights.shape == task_weights_device.shape:
                metric.task_weights.copy_(task_weights_device)

    if args.select_best_hf:
        checkpoint_callback = ModelCheckpoint(save_top_k=-1, filename="epoch-{epoch:03d}")
    else:
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                              filename="best")
    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", mode="min",
                      patience=args.patience),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir=args.save_dir,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)

    val_truths_for_select = np.stack([dp.y for dp in val_pts])
    if args.select_best_hf:
        best_path = select_checkpoint_by_val_hf(
            trainer, model, val_loader, val_truths_for_select, checkpoint_callback.dirpath,
        ) or trainer.checkpoint_callback.best_model_path
    else:
        best_path = trainer.checkpoint_callback.best_model_path
    print(f"\nbest checkpoint: {best_path}")

    pred_batches = trainer.predict(model, test_loader, ckpt_path=best_path, weights_only=False)
    test_preds = torch.cat(pred_batches).cpu().numpy()
    test_truths = np.stack([dp.y for dp in test_pts])

    val_pred_batches = trainer.predict(model, val_loader, ckpt_path=best_path, weights_only=False)
    val_preds = torch.cat(val_pred_batches).cpu().numpy()
    val_truths = val_truths_for_select

    if args.atomref_hf:
        test_preds = add_hf_baseline(test_preds, test_hf_base)
        test_truths = add_hf_baseline(test_truths, test_hf_base)
        val_preds = add_hf_baseline(val_preds, val_hf_base)
        val_truths = add_hf_baseline(val_truths, val_hf_base)

    raw_test_preds = test_preds.copy()
    raw_val_preds = val_preds.copy()
    hf_calibration = np.array([1.0, 0.0], dtype=np.float32)
    if args.calibrate_hf_affine:
        slope, intercept = fit_affine_calibration(val_preds, val_truths, 0)
        hf_calibration = np.array([slope, intercept], dtype=np.float32)
        val_preds = apply_target_affine(val_preds, 0, slope, intercept)
        test_preds = apply_target_affine(test_preds, 0, slope, intercept)
        print(f"Hf affine calibration from val: y = {slope:.6f} * pred + {intercept:.6f}")

    mae = np.abs(test_preds - test_truths).mean(axis=0)
    rmse = np.sqrt(((test_preds - test_truths) ** 2).mean(axis=0))

    print("\ntest set per-target metrics:")
    for name, m, r in zip(TARGET_COLS, mae, rmse):
        print(f"  {name:>8s}  MAE={m:.4f}  RMSE={r:.4f}")

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "predictions.npz",
             test_preds=test_preds, test_truths=test_truths,
             val_preds=val_preds, val_truths=val_truths,
             raw_test_preds=raw_test_preds, raw_val_preds=raw_val_preds,
             hf_calibration=hf_calibration,
             atomref_nums=atomref_nums, atomref_coef=atomref_coef,
             target_names=np.array(TARGET_COLS),
             seed=args.seed, test_seed=args.test_seed)


if __name__ == "__main__":
    main()