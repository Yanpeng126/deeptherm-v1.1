# Hf-focused 10-model ensemble

This recipe reproduces the heterogeneous same-architecture ensemble used for the Hf-focused run.
It follows the ensemble description in Section 2.3: ten DeepTherm models with random-seed and hyperparameter variation, fixed test set, and inverse validation-MAE weighting.

Required external file:

```text
best-qm9-pretrain-v1.ckpt
```

Place the checkpoint in the repository root, then run:

```bash
PYTHON=.venv/Scripts/python ./run_hf2118_ensemble.sh
```

Expected random-split ensemble metrics from the recorded run:

```text
Hf_298   MAE=2.1178  RMSE=5.8482
S_298    MAE=1.4695  RMSE=2.0395
Cp_300   MAE=0.7707  RMSE=1.0868
Cp_400   MAE=0.7614  RMSE=1.1272
Cp_500   MAE=0.7389  RMSE=1.1428
Cp_600   MAE=0.7186  RMSE=1.1360
Cp_800   MAE=0.7078  RMSE=1.1263
Cp_1000  MAE=0.7224  RMSE=1.1037
Cp_1500  MAE=0.7180  RMSE=1.0805
```

## n-Heptane mechanism prediction

The prediction workflow requires Supplementary Data 4 and Supplementary Data 5.
These source files and the trained checkpoints are not included in the repository.

First reproduce or provide the ten run directories under
`runs/hf2118_mixed_ensemble`. Then run:

```bash
PYTHON=.venv/Scripts/python ./run_nheptane_prediction.sh \
    "/path/to/Supplementary Data 4_predicted dataset.xlsx" \
    "/path/to/Supplementary Data 5_NASA polynomial thermochemistry in n-heptane mechanism.txt"
```

The script extracts the unique mechanism species from Data 5, matches them to
the Data 4 SMILES and thermochemistry values, and applies the same ten-model
inverse validation-MAE ensemble. The default output is:

```text
results/nheptane_comparison.xlsx
```

The workbook contains `Statistics`, `Comparison`, and `Ensemble` worksheets.
