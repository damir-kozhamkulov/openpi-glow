"""Model config for GLOW: pi05 that predicts its plan state (PyTorch path only)."""

import dataclasses

import safetensors.torch

from openpi.models import pi0_config


@dataclasses.dataclass(frozen=True)
class GlowPi0Config(pi0_config.Pi0Config):
    # Weight (lambda) of the plan cross-entropy term. 0 trains the flow loss alone, the plan block
    # still in the prefix as teacher-forced conditioning.
    subtask_ce_weight: float = 0.0
    # Plan bottleneck: the action expert reads language only through a second copy of the plan
    # block (the "action block") that sees nothing but the images, and the images no longer read
    # the prompt. The prompt and the first block (the planner) are unchanged, so the planner still
    # predicts the plan from the task string; the actions cannot see the task string at all. The
    # data side must append the action block (LeRobotLiberoGlowDataConfig.plan_bottleneck).
    plan_bottleneck: bool = False

    def create(self, rng):
        raise NotImplementedError("GLOW trains on the PyTorch path only (scripts/train_pytorch.py)")

    def create_pytorch(self):
        from openpi.models_pytorch import glow_pi0_pytorch

        return glow_pi0_pytorch.GlowPI0Pytorch(self)

    def load_pytorch(self, train_config, weight_path: str):
        model = self.create_pytorch()
        safetensors.torch.load_model(model, weight_path)
        return model
