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
    # (grounding), the subgoal_operation text, the object / target nouns and the atom text.
    module_spatial: bool = True
    module_temporal: bool = True
    module_grounding: bool = True
    module_subgoal_text: bool = True
    module_nouns: bool = True
    module_atom: bool = True
    # The Temporal active slot reads the frames since its stage began (the data config's
    # module_stage_clock must match: inference then needs the client's plan_stage_frames).
    module_stage_clock: bool = False
    # Encoder width and transformer depth.
    module_width: int = 512
    module_depth: int = 2
    # Each encoder token = its slot's pooled text embedding + a residual whose last layer starts at
    # zero (module_encoders), so a fresh encoder emits language-like tokens, not random ones.
    module_text_anchor: bool = False
    # The anchor of each token: "mean" = its text's mean-pooled token embedding, "head" = one head
    # word's embedding (module_inputs.head_words).
    module_anchor: str = "mean"
    # With "head" anchors: the Spatial summary (CTX) anchored to the active stage's predicate word
    # (True), or to the mean of the shown stages' (False: the Spatial tokens then say nothing about
    # which stage is active); the Temporal active token anchored to the active stage's "ordinal"
    # word or its "object" head noun (module_encoders.TemporalEncoder).
    module_ctx_active: bool = True
    module_temporal_active_anchor: str = "ordinal"
    # Above 0: the Temporal encoder scales each Spatial group in the prefix, starting at 1 for the
    # active stage and at this for the others (module_encoders.TemporalEncoder).
    module_temporal_gate: float = 0.0
    # Above 0: each Spatial group enters the prefix at a constant, the active stage's at 1 and the
    # others' at this; the stage pointer alone decides which. Excludes module_temporal_gate.
    module_stage_gate: float = 0.0
    # The Temporal tokens are masked out of the prefix, like padding.
    module_temporal_masked: bool = False
    # The Temporal encoder reads only the active stage's runtime key, nothing else of the plan
    # (module_encoders.TemporalEncoder); excludes the stage clock and the gate.
    module_temporal_key_only: bool = False
    # The size each encoder token enters the prefix at: "language" = the RMS of the sample's
    # language tokens (with the task string dropped, only BOS and the newline answer token);
    # "words" = the mean RMS of the sample's own module-text tokens as the prefix embeds them.
    module_token_scale: str = "language"
    # The encoders start from scratch while the backbone is fine-tuned: their learning rate is the
    # schedule's times this.
    encoder_lr_scale: float = 16.0
    # Training loss = module_flow_weight x flow + module_distill_weight x distillation, where the
    # distillation target is the velocity the same model predicts from the task string instead of
    # the module tokens (a teacher pass without gradients, on the same images, state, noise and
    # time). The task string reaches the model only for that pass, tokenized to
    # module_teacher_prompt_len; inference never sees it. A training sample may carry a flow weight
    # (glow_flow_weight, 0 for counterfactual samples whose demo action belongs to another task):
    # the flow term is then the weighted mean over the batch.
    module_flow_weight: float = 1.0
    module_distill_weight: float = 0.0
    module_teacher_prompt_len: int = 48
    # Train only the encoders: every pretrained parameter is frozen, so the teacher pass is the
    # fine-tuning init itself.
    freeze_pretrained: bool = False
    # The progress monitor decides the active stage (module_encoders.ProgressMonitor): a done
    # probability per stage from the plan's runtime keys and the prefix's camera tokens, the first
    # stage not done is active, and the Spatial groups enter at soft_stage_gates with
    # module_stage_gate as the other stages' size. Needs the Spatial encoder and module_stage_gate,
    # replaces the Temporal encoder (module_temporal off). An inference sample whose client gives
    # the stage (oracle_plan_index) enters at the fixed gates instead, as an ablation.
    module_temporal_monitor: bool = False
    module_monitor_width: int = 512
    module_monitor_depth: int = 2
    module_monitor_heads: int = 8
    # Camera tokens are averaged over pool x pool patches before the monitor reads them.
    module_monitor_pool: int = 2
    # Training: loss += module_done_weight x the done BCE; with probability module_monitor_swap_p a
    # sample's done BCE is also taken on its plan with the stages permuted (the labels follow
    # their keys), so the monitor reads the keys, not the stage positions; with probability
    # module_monitor_teacher_forcing a sample's gates come from its labelled stage instead of the
    # monitor.
    module_done_weight: float = 1.0
    module_monitor_swap_p: float = 0.3
    module_monitor_teacher_forcing: float = 0.5
    # A trained monitor (directory with model.safetensors, parameters under glow_progress_monitor.)
    # loaded over the init weights before training, e.g. scripts/glow_progress_monitor.py --save-dir.
    module_monitor_init: str | None = None
    # Freeze the Spatial encoder too (with freeze_pretrained only the monitor then trains).
    module_freeze_spatial: bool = False

    def create(self, rng):
        raise NotImplementedError("GLOW trains on the PyTorch path only (scripts/train_pytorch.py)")

    def create_pytorch(self):
        from openpi.models_pytorch import glow_module_pi0_pytorch

        return glow_module_pi0_pytorch.GlowModulePI0Pytorch(self)

    def load_pytorch(self, train_config, weight_path: str):
        """Strict, except that a monitor row also takes a checkpoint with no monitor (its monitor
        then has to come from elsewhere: `model.monitor_missing`) and with a Temporal encoder it
        does not use."""
        model = self.create_pytorch()
        if not self.module_temporal_monitor:
            safetensors.torch.load_model(model, weight_path)
            return model
        missing, unexpected = safetensors.torch.load_model(model, weight_path, strict=False)
        model.check_partial_load(weight_path, missing, unexpected)
        return model
