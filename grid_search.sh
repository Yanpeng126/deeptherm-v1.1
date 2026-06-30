#!/bin/bash
set -e

CSV=${CSV:-data/curated_4997.csv}
OUT_ROOT=${OUT_ROOT:-runs/grid}
SEED=${SEED:-42}
mkdir -p "$OUT_ROOT"

for bs in 32 64; do
for dh in 300 600; do
for dp in 3 5; do
    name="bs${bs}_dh${dh}_dp${dp}"
    if [ -f "$OUT_ROOT/$name/predictions.npz" ]; then
        echo "=== $name already done, skipping ==="
        continue
    fi
    echo "=== $name ==="
    python src/train.py \
        --csv "$CSV" \
        --save-dir "$OUT_ROOT/$name" \
        --epochs 300 \
        --patience 300 \
        --batch-size "$bs" \
        --d-hidden "$dh" \
        --depth "$dp" \
        --ecfp-bits 1024 \
        --seed "$SEED"
done
done
done

echo "=== aggregating ==="
python src/grid_aggregate.py --runs "$OUT_ROOT"/*/