"""Model configs for GLOW (PyTorch path only): pi05 that predicts its plan state, and pi05 that
reads the planning module through learned Spatial and Temporal encoders."""

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


@dataclasses.dataclass(frozen=True)
class GlowModulePi0Config(pi0_config.Pi0Config):
    # Which encoders feed the prefix (at least one), and the Spatial encoder's inputs: the anchors
    # (grounding) and the subgoal_operation text.
    module_spatial: bool = True
    module_temporal: bool = True
    module_grounding: bool = True
    module_subgoal_text: bool = True
    # Encoder width and transformer depth.
    module_width: int = 512
    module_depth: int = 2
    # Each encoder token = its slot's pooled text embedding + a residual whose last layer starts at
    # zero (module_encoders), so a fresh encoder emits language-like tokens, not random ones.
    module_text_anchor: bool = False
    # The encoders start from scratch while the backbone is fine-tuned: their learning rate is the
    # schedule's times this.
    encoder_lr_scale: float = 16.0
    # Training loss = module_flow_weight x flow + module_distill_weight x distillation, where the
    # distillation target is the velocity the same model predicts from the task string instead of
    # the module tokens (a teacher pass without gradients, on the same images, state, noise and
    # time). The task string reaches the model only for that pass, tokenized to
    # module_teacher_prompt_len; inference never sees it.
    module_flow_weight: float = 1.0
    module_distill_weight: float = 0.0
    module_teacher_prompt_len: int = 48
    # Train only the encoders: every pretrained parameter is frozen, so the teacher pass is the
    # fine-tuning init itself.
    freeze_pretrained: bool = False

    def create(self, rng):
        raise NotImplementedError("GLOW trains on the PyTorch path only (scripts/train_pytorch.py)")

    def create_pytorch(self):
        from openpi.models_pytorch import glow_module_pi0_pytorch

        return glow_module_pi0_pytorch.GlowModulePI0Pytorch(self)

    def load_pytorch(self, train_config, weight_path: str):
        model = self.create_pytorch()
        safetensors.torch.load_model(model, weight_path)
        return model
