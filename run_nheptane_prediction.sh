#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "usage: $0 DATA4_XLSX DATA5_TXT [RUNS_ROOT] [OUTPUT_XLSX]"
    exit 1
fi

PYTHON=${PYTHON:-python}
DATA4=$1
DATA5=$2
RUNS_ROOT=${3:-runs/hf2118_mixed_ensemble}
OUTPUT=${4:-results/nheptane_comparison.xlsx}

"$PYTHON" src/predict_nheptane.py \
    --data4 "$DATA4" \
    --data5 "$DATA5" \
    --runs-root "$RUNS_ROOT" \
    --out "$OUTPUT"
