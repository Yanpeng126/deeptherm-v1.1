#!/bin/bash
set -e

CSV=${CSV:-data/curated_4997.csv}
SPLIT_MODE=${SPLIT_MODE:-random}
OUT_ROOT=${OUT_ROOT:-runs/ensemble_${SPLIT_MODE}}
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}

mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
    echo "=== seed=$seed  split=$SPLIT_MODE ==="
    python src/train.py \
        --csv "$CSV" \
        --save-dir "$OUT_ROOT/seed_$seed" \
        --epochs 300 \
        --patience 300 \
        --batch-size 32 \
        --d-hidden 600 \
        --depth 5 \
        --ecfp-bits 1024 \
        --split-mode "$SPLIT_MODE" \
        --seed "$seed" \
        --test-seed 42
done

echo "=== combining ==="
python src/ensemble.py \
    --runs "$OUT_ROOT"/seed_* \
    --out "$OUT_ROOT/ensemble_predictions.npz"