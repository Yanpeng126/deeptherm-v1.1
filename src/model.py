from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
from chemprop.nn.transforms import UnscaleTransform


class ECFPProjection(nn.Module):
    """Random projection of Morgan fingerprints into a low-dimensional embedding.

    Weights are frozen at random initialization and the output is rescaled by a
    fixed small factor so that the fingerprint signal is preserved in the
    forward pass without dominating gradient updates from the GNN branch.
    """

    def __init__(self, n_bits: int, d_out: int, scale: float = 0.01):
        super().__init__()
        self.n_bits = n_bits
        self.d_out = d_out
        self.proj = nn.Linear(n_bits, d_out)
        for p in self.proj.parameters():
            p.requires_grad = False
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * torch.relu(self.proj(x))


class BondAttentionEncoder(nn.Module):
    """DMPNN encoder with bond-level attention before bond-to-atom aggregation.

    Implements the global attention mechanism described in Section 2.1: bond
    hidden states from DMPNN message passing are weighted by an attention score
    a_ij computed within each molecule, then summed at each target atom to
    produce the attended atom representation h^att_i.
    """

    def __init__(self, d_h: int = 300, depth: int = 3, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self._d_h = d_h
        self._depth = depth
        self._num_heads = num_heads
        self._dropout = dropout
        self.mp = BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout,
                                     undirected=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @property
    def output_dim(self) -> int:
        return self.mp.output_dim

    @property
    def V_d_transform(self):
        return self.mp.V_d_transform

    @property
    def graph_transform(self):
        return self.mp.graph_transform

    @property
    def hparams(self):
        # Return a fresh dict on every access; chemprop's MPNN._load pops
        # "cls" from this dict during checkpoint loading, and a stale
        # reference would break subsequent accesses.
        return {
            "cls": self.__class__,
            "d_h": self._d_h,
            "depth": self._depth,
            "num_heads": self._num_heads,
            "dropout": self._dropout,
        }

    def forward(self, bmg, V_d: Tensor | None = None) -> Tensor:
        bmg = self.mp.graph_transform(bmg)
        H_0 = self.mp.initialize(bmg)
        H_bond = self.mp.tau(H_0)
        for _ in range(1, self.mp.depth):
            if self.mp.undirected:
                H_bond = (H_bond + H_bond[bmg.rev_edge_index]) / 2
            M = self.mp.message(H_bond, bmg)
            H_bond = self.mp.update(M, H_0)

        n_mols = int(bmg.batch.max().item()) + 1
        d = H_bond.shape[1]
        device = H_bond.device

        if H_bond.shape[0] == 0:
            M = torch.zeros(len(bmg.V), d, device=device, dtype=H_bond.dtype)
            return self.mp.finalize(M, bmg.V, V_d)

        bond_to_mol = bmg.batch[bmg.edge_index[0]]
        counts = torch.bincount(bond_to_mol, minlength=n_mols)
        max_n = int(counts.max().item())

        starts = torch.zeros(n_mols, dtype=torch.long, device=device)
        starts[1:] = counts.cumsum(0)[:-1]
        local_idx = torch.arange(H_bond.shape[0], device=device) - starts[bond_to_mol]

        H_pad = H_bond.new_zeros(n_mols, max_n, d)
        H_pad[bond_to_mol, local_idx] = H_bond

        pad_mask = torch.ones(n_mols, max_n, dtype=torch.bool, device=device)
        pad_mask[bond_to_mol, local_idx] = False

        has_bond = counts > 0
        if has_bond.all():
            H_attn, _ = self.attn(H_pad, H_pad, H_pad,
                                  key_padding_mask=pad_mask,
                                  need_weights=False)
        else:
            valid = has_bond.nonzero(as_tuple=True)[0]
            H_attn_valid, _ = self.attn(H_pad[valid], H_pad[valid], H_pad[valid],
                                        key_padding_mask=pad_mask[valid],
                                        need_weights=False)
            H_attn = H_pad.new_zeros(H_pad.shape)
            H_attn[valid] = H_attn_valid

        H_attended = H_attn[bond_to_mol, local_idx]

        index = bmg.edge_index[1].unsqueeze(1).repeat(1, d)
        M = torch.zeros(len(bmg.V), d, dtype=H_attended.dtype, device=device)
        M.scatter_reduce_(0, index, H_attended, reduce="sum",
                          include_self=False)

        return self.mp.finalize(M, bmg.V, V_d)


def build_deeptherm(
    n_targets: int,
    d_hidden: int = 300,
    depth: int = 3,
    num_heads: int = 4,
    ffn_hidden: int = 300,
    ffn_layers: int = 1,
    dropout: float = 0.0,
    ecfp_bits: int = 0,
    ecfp_proj_dim: int = 64,
    output_transform=None,
) -> MPNN:
    encoder = BondAttentionEncoder(
        d_h=d_hidden,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    )
    agg = NormAggregation(norm=100.0)

    if ecfp_bits > 0:
        X_d_transform = ECFPProjection(ecfp_bits, ecfp_proj_dim, scale=0.01)
        predictor_input_dim = d_hidden + ecfp_proj_dim
    else:
        X_d_transform = None
        predictor_input_dim = d_hidden

    predictor = RegressionFFN(
        n_tasks=n_targets,
        input_dim=predictor_input_dim,
        hidden_dim=ffn_hidden,
        n_layers=ffn_layers,
        dropout=dropout,
        output_transform=output_transform,
    )
    return MPNN(
        message_passing=encoder,
        agg=agg,
        predictor=predictor,
        X_d_transform=X_d_transform,
    )


def load_deeptherm(
    ckpt_path: str | Path,
    n_targets: int = 9,
    d_hidden: int = 600,
    depth: int = 5,
    num_heads: int = 4,
    ffn_hidden: int = 300,
    ffn_layers: int = 1,
    ecfp_bits: int = 1024,
    ecfp_proj_dim: int = 64,
    map_location: str = "cpu",
) -> MPNN:
    """Load a trained DeepTherm model from a Lightning checkpoint.

    Prefer this over ``MPNN.load_from_checkpoint`` for checkpoints trained by
    older versions of the code (where the encoder incorrectly reported itself
    as a plain BondMessagePassing in its hparams). The hyperparameters given
    here must match the ones used at training time; the values in the default
    arguments correspond to the ensemble configuration shipped with this repo.

    Returns a model in eval mode with predictor.output_transform restored, so
    predictions come back in physical units (kcal/mol for enthalpy, cal/mol/K
    for entropy and heat capacity).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = ckpt["state_dict"]

    # Placeholder output_transform. The real mean and scale are overwritten
    # from the checkpoint state_dict during load_state_dict below.
    placeholder_output = UnscaleTransform(
        np.zeros(n_targets, dtype=np.float32),
        np.ones(n_targets, dtype=np.float32),
    )

    model = build_deeptherm(
        n_targets=n_targets,
        d_hidden=d_hidden,
        depth=depth,
        num_heads=num_heads,
        ffn_hidden=ffn_hidden,
        ffn_layers=ffn_layers,
        ecfp_bits=ecfp_bits,
        ecfp_proj_dim=ecfp_proj_dim,
        output_transform=placeholder_output,
    )

    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys when loading {ckpt_path}: "
            f"{result.unexpected_keys[:5]}..."
        )
    real_missing = [k for k in result.missing_keys
                    if not k.endswith(".running_mean")
                    and not k.endswith(".running_var")
                    and not k.endswith(".num_batches_tracked")]
    if real_missing:
        raise RuntimeError(
            f"Missing keys when loading {ckpt_path}: {real_missing[:5]}..."
        )

    model.eval()
    return model