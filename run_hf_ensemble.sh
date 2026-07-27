#!/bin/bash
set -e

CSV=${CSV:-data/curated_4997.csv}
SPLIT_MODE=${SPLIT_MODE:-random}
OUT_ROOT=${OUT_ROOT:-runs/hf_ensemble_${SPLIT_MODE}}
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}
BATCH_SIZE=${BATCH_SIZE:-32}
D_HIDDEN=${D_HIDDEN:-600}
DEPTH=${DEPTH:-5}
FFN_HIDDEN=${FFN_HIDDEN:-300}
FFN_LAYERS=${FFN_LAYERS:-1}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-150}
HF_LOSS_WEIGHT=${HF_LOSS_WEIGHT:-20}
PYTHON=${PYTHON:-python}

mkdir -p "$OUT_ROOT"

echo "split=$SPLIT_MODE bs=$BATCH_SIZE d_hidden=$D_HIDDEN depth=$DEPTH hf_weight=$HF_LOSS_WEIGHT"

for seed in $SEEDS; do
    echo "=== seed=$seed ==="
    $PYTHON src/train.py \
        --csv "$CSV" \
        --save-dir "$OUT_ROOT/seed_$seed" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch-size "$BATCH_SIZE" \
        --d-hidden "$D_HIDDEN" \
        --depth "$DEPTH" \
        --ffn-hidden "$FFN_HIDDEN" \
        --ffn-layers "$FFN_LAYERS" \
        --ecfp-bits 1024 \
        --ecfp-mode projected \
        --hf-loss-weight "$HF_LOSS_WEIGHT" \
        --calibrate-hf-affine \
        --split-mode "$SPLIT_MODE" \
        --seed "$seed" \
        --test-seed 42
done

echo "=== combining ==="
$PYTHON src/ensemble.py \
    --runs "$OUT_ROOT"/seed_* \
    --out "$OUT_ROOT/ensemble_predictions.npz" \
    --calibrate-hf-affine