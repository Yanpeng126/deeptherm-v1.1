#!/bin/bash
set -e

CSV=${CSV:-data/curated_4997.csv}
PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-runs/hf2118_mixed_ensemble}
CKPT=${CKPT:-best-qm9-pretrain-v1.ckpt}

mkdir -p "$OUT_ROOT"

run_scratch() {
    name=$1
    seed=$2
    hf_weight=$3
    epochs=$4
    "$PYTHON" src/train.py \
        --csv "$CSV" \
        --save-dir "$OUT_ROOT/$name" \
        --epochs "$epochs" \
        --patience "$epochs" \
        --batch-size 32 \
        --d-hidden 600 \
        --depth 5 \
        --ffn-hidden 300 \
        --ffn-layers 1 \
        --ecfp-bits 1024 \
        --ecfp-mode projected \
        --hf-loss-weight "$hf_weight" \
        --calibrate-hf-affine \
        --split-mode random \
        --seed "$seed" \
        --test-seed 42
}

run_transfer() {
    name=$1
    seed=$2
    ffn_layers=$3
    epochs=$4
    "$PYTHON" src/run_transfer.py \
        --csv "$CSV" \
        --save-dir "$OUT_ROOT/$name" \
        --pretrained-ckpt "$CKPT" \
        --freeze-encoder-epochs 5 \
        --encoder-grad-scale 1.0 \
        --epochs "$epochs" \
        --patience "$epochs" \
        --batch-size 32 \
        --d-hidden 600 \
        --depth 5 \
        --ffn-hidden 300 \
        --ffn-layers "$ffn_layers" \
        --ecfp-bits 1024 \
        --ecfp-mode projected \
        --hf-loss-weight 20 \
        --calibrate-hf-affine \
        --split-mode random \
        --seed "$seed" \
        --test-seed 42
}

run_scratch exp04_legacy600_w5_e150 42 5 150
run_scratch exp06_legacy600_w20_e150 42 20 150
run_scratch exp09_legacy600_w20_e150_seed44 44 20 150
run_scratch exp_w20_e150_seed_46 46 20 150
run_scratch exp_w20_e150_seed_47 47 20 150
run_scratch exp_w20_e150_seed_48 48 20 150
run_scratch exp_w20_e150_seed_49 49 20 150
run_scratch exp_w20_e150_seed_50 50 20 150
run_transfer transfer_qm9_seed42_w20_f2_e150 42 2 150
run_transfer transfer_qm9_seed42_w20_f3_e120 42 3 120

"$PYTHON" src/ensemble.py \
    --runs \
    "$OUT_ROOT/exp04_legacy600_w5_e150" \
    "$OUT_ROOT/exp06_legacy600_w20_e150" \
    "$OUT_ROOT/exp09_legacy600_w20_e150_seed44" \
    "$OUT_ROOT/exp_w20_e150_seed_46" \
    "$OUT_ROOT/exp_w20_e150_seed_47" \
    "$OUT_ROOT/exp_w20_e150_seed_48" \
    "$OUT_ROOT/exp_w20_e150_seed_49" \
    "$OUT_ROOT/exp_w20_e150_seed_50" \
    "$OUT_ROOT/transfer_qm9_seed42_w20_f2_e150" \
    "$OUT_ROOT/transfer_qm9_seed42_w20_f3_e120" \
    --out "$OUT_ROOT/ensemble_predictions.npz"