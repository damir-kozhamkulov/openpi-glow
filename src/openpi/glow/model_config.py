"""Model config for GLOW: pi05 plus the subtask cross-entropy head (PyTorch path only)."""

import dataclasses

import safetensors.torch

from openpi.models import pi0_config


@dataclasses.dataclass(frozen=True)
class GlowPi0Config(pi0_config.Pi0Config):
    # Weight of the subtask cross-entropy term. 0 skips the CE pass entirely, so a row with
    # subtask_ce_weight=0 trains the stock flow objective on the plan-clause prompts.
    subtask_ce_weight: float = 0.0

    def create(self, rng):
        raise NotImplementedError("GLOW trains on the PyTorch path only (scripts/train_pytorch.py)")

    def create_pytorch(self):
        from openpi.models_pytorch import glow_pi0_pytorch

        return glow_pi0_pytorch.GlowPI0Pytorch(self)

    def load_pytorch(self, train_config, weight_path: str):
        model = self.create_pytorch()
        safetensors.torch.load_model(model, weight_path)
        return model
