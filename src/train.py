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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=Path("runs/default"))
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
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
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
        output_transform=output_transform,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                        filename="best"),
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
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\nbest checkpoint: {best_path}")

    pred_batches = trainer.predict(model, test_loader, ckpt_path=best_path)
    test_preds = torch.cat(pred_batches).cpu().numpy()
    test_truths = np.stack([dp.y for dp in test_pts])

    val_pred_batches = trainer.predict(model, val_loader, ckpt_path=best_path)
    val_preds = torch.cat(val_pred_batches).cpu().numpy()
    val_truths = np.stack([dp.y for dp in val_pts])

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
             target_names=np.array(TARGET_COLS),
             seed=args.seed, test_seed=args.test_seed)


if __name__ == "__main__":
    main()