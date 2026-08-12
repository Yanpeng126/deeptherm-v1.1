from __future__ import annotations

import argparse
import os
import runpy
import sys

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


ROOT = os.path.abspath(os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


def pop_transfer_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pretrained-ckpt", default="best-qm9-pretrain-v1.ckpt")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--encoder-grad-scale", type=float, default=0.2)
    known, remaining = parser.parse_known_args(argv[1:])
    return known, [argv[0], *remaining]


def add_default_arg(argv, flag, value):
    if flag not in argv:
        argv.extend([flag, str(value)])


def clean_key(key):
    return key.replace("network.", "", 1)


def is_transferable(key, value, target_state, encoder_only):
    if key.startswith("metrics."):
        return False
    if key.startswith("predictor.criterion."):
        return False
    if encoder_only and not key.startswith("message_passing."):
        return False
    if key not in target_state:
        return False
    return target_state[key].shape == value.shape


class TransferHook(Callback):
    def __init__(self, ckpt_path, freeze_encoder_epochs=3, encoder_only=False, encoder_grad_scale=0.2):
        self.ckpt_path = ckpt_path
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_only = encoder_only
        self.encoder_grad_scale = encoder_grad_scale
        self.loaded = False

    def on_fit_start(self, trainer, pl_module):
        if self.loaded:
            return
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        source_state = checkpoint.get("state_dict", checkpoint)
        target_state = pl_module.state_dict()

        transfer_state = {}
        for key, value in source_state.items():
            key = clean_key(key)
            if is_transferable(key, value, target_state, self.encoder_only):
                transfer_state[key] = value

        result = pl_module.load_state_dict(transfer_state, strict=False)
        print(f"transfer_ckpt={self.ckpt_path}")
        print(f"transfer_tensors={len(transfer_state)}")
        print(f"missing_tensors={len(result.missing_keys)}")
        self.loaded = True

    def on_train_epoch_start(self, trainer, pl_module):
        encoder = getattr(pl_module, "message_passing", None)
        if encoder is None:
            return
        freeze = trainer.current_epoch < self.freeze_encoder_epochs
        for param in encoder.parameters():
            param.requires_grad = not freeze
        if trainer.current_epoch in (0, self.freeze_encoder_epochs):
            state = "frozen" if freeze else "trainable"
            print(f"encoder={state}")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.encoder_grad_scale == 1.0:
            return
        encoder = getattr(pl_module, "message_passing", None)
        if encoder is None or trainer.current_epoch < self.freeze_encoder_epochs:
            return
        for param in encoder.parameters():
            if param.grad is not None:
                param.grad.mul_(self.encoder_grad_scale)


transfer_args, train_argv = pop_transfer_args(sys.argv)
add_default_arg(train_argv, "--d-hidden", 600)
add_default_arg(train_argv, "--depth", 5)
add_default_arg(train_argv, "--ecfp-bits", 1024)
add_default_arg(train_argv, "--ffn-hidden", 300)
add_default_arg(train_argv, "--ffn-layers", 2)
add_default_arg(train_argv, "--dropout", 0.2)

real_trainer_init = pl.Trainer.__init__


def trainer_init(self, *args, **kwargs):
    callbacks = list(kwargs.get("callbacks") or [])
    callbacks.append(TransferHook(
        transfer_args.pretrained_ckpt,
        freeze_encoder_epochs=transfer_args.freeze_encoder_epochs,
        encoder_only=transfer_args.encoder_only,
        encoder_grad_scale=transfer_args.encoder_grad_scale,
    ))
    kwargs["callbacks"] = callbacks
    real_trainer_init(self, *args, **kwargs)


pl.Trainer.__init__ = trainer_init
sys.argv = train_argv
sys.argv[0] = "./src/train.py"
runpy.run_path("./src/train.py", run_name="__main__")
