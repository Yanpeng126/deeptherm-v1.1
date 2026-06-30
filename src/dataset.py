from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer


TARGET_COLS = [
    "Hf_298", "S_298",
    "Cp_300", "Cp_400", "Cp_500", "Cp_600",
    "Cp_800", "Cp_1000", "Cp_1500",
]


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_datapoints(csv_path: str | Path,
                    ecfp_bits: int = 0,
                    ecfp_radius: int = 2) -> list[MoleculeDatapoint]:
    df = pd.read_csv(csv_path)
    points: list[MoleculeDatapoint] = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        y = row[TARGET_COLS].to_numpy(dtype=np.float32)
        if ecfp_bits > 0:
            x_d = morgan_fp(smi, ecfp_radius, ecfp_bits)
            points.append(MoleculeDatapoint.from_smi(smi, y=y, x_d=x_d))
        else:
            points.append(MoleculeDatapoint.from_smi(smi, y=y))
    return points


def split_with_fixed_test(
    points: list[MoleculeDatapoint],
    train_val_seed: int,
    test_seed: int = 42,
    test_frac: float = 0.10,
    val_in_rest_frac: float = 0.10,
):
    """Hold out a fixed test set with `test_seed`; split the remainder into
    train and val using `train_val_seed`. Calling this with different
    `train_val_seed` values produces different train/val splits while keeping
    the test set identical, which is what ensemble averaging requires.
    """
    n_total = len(points)

    rng_test = np.random.default_rng(test_seed)
    idx = rng_test.permutation(n_total)
    n_test = int(n_total * test_frac)
    test_idx = idx[:n_test]
    rest_idx = idx[n_test:]

    rng_tv = np.random.default_rng(train_val_seed)
    rest_shuffled = rest_idx[rng_tv.permutation(len(rest_idx))]
    n_val = int(len(rest_idx) * val_in_rest_frac)
    val_idx = rest_shuffled[:n_val]
    train_idx = rest_shuffled[n_val:]

    train = [points[i] for i in train_idx]
    val = [points[i] for i in val_idx]
    test = [points[i] for i in test_idx]
    return train, val, test


def split_by_complexity(
    points: list[MoleculeDatapoint],
    train_val_seed: int,
    test_frac: float = 0.10,
    val_in_rest_frac: float = 0.10,
):
    """Hierarchical complexity-based split (Section 2.5): rank molecules by
    heavy atom count and reserve the largest `test_frac` as the test set.
    The remaining smaller molecules are split into train and val by
    `train_val_seed`. The test set is identical across ensemble runs.
    """
    heavy_counts = np.array([dp.mol.GetNumHeavyAtoms() for dp in points])
    sorted_idx = np.argsort(heavy_counts, kind="stable")

    n_total = len(points)
    n_test = int(n_total * test_frac)
    rest_idx = sorted_idx[:-n_test]
    test_idx = sorted_idx[-n_test:]

    rng_tv = np.random.default_rng(train_val_seed)
    rest_shuffled = rest_idx[rng_tv.permutation(len(rest_idx))]
    n_val = int(len(rest_idx) * val_in_rest_frac)
    val_idx = rest_shuffled[:n_val]
    train_idx = rest_shuffled[n_val:]

    train = [points[i] for i in train_idx]
    val = [points[i] for i in val_idx]
    test = [points[i] for i in test_idx]
    return train, val, test


def make_dataset(points: list[MoleculeDatapoint]) -> MoleculeDataset:
    return MoleculeDataset(points, featurizer=SimpleMoleculeMolGraphFeaturizer())