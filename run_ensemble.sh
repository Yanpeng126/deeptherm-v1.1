#!/bin/bash
set -e

CSV=${CSV:-data/curated_4997.csv}
SPLIT_MODE=${SPLIT_MODE:-random}
OUT_ROOT=${OUT_ROOT:-runs/ensemble_${SPLIT_MODE}}
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}

# The two split modes prefer different hyperparameters. The larger model
# tuned on random splits overfits to smaller training molecules and
# generalizes worse to larger ones under complexity-based extrapolation,
# so the default chemprop-scale configuration is used for the complexity-based.
if [ "$SPLIT_MODE" = "complexity" ]; then
    BATCH_SIZE=${BATCH_SIZE:-64}
    D_HIDDEN=${D_HIDDEN:-300}
    DEPTH=${DEPTH:-3}
else
    BATCH_SIZE=${BATCH_SIZE:-32}
    D_HIDDEN=${D_HIDDEN:-600}
    DEPTH=${DEPTH:-5}
fi

echo "split=$SPLIT_MODE  bs=$BATCH_SIZE  d_hidden=$D_HIDDEN  depth=$DEPTH"

mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
    echo "=== seed=$seed ==="
    python src/train.py \
        --csv "$CSV" \
        --save-dir "$OUT_ROOT/seed_$seed" \
        --epochs 300 \
        --patience 300 \
        --batch-size "$BATCH_SIZE" \
        --d-hidden "$D_HIDDEN" \
        --depth "$DEPTH" \
        --ecfp-bits 1024 \
        --split-mode "$SPLIT_MODE" \
        --seed "$seed" \
        --test-seed 42
done

echo "=== combining ==="
python src/ensemble.py \
    --runs "$OUT_ROOT"/seed_* \
    --out "$OUT_ROOT/ensemble_predictions.npz"