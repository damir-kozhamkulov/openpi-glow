"""GLOW: the planning module as learned prefix tokens on top of pi05.

- plan_pack: the per-task plan chain and per-episode stage ranges (built by
  glow_repo/scripts/build_plan_pack.py).
- module_inputs: the data transform that turns a sample's plan into the encoders' arrays.
- module_encoders: the Spatial and Temporal encoders and the progress monitor.
- model_config: the model config (GlowModulePi0Config).
"""
