"""GLOW: plan-state conditioning and subtask prediction on top of pi05.

- plan_pack: the per-task plan chain and per-episode stage ranges (built by
  glow_repo/scripts/build_plan_pack.py).
- plan_prompt: data transforms that write the plan clause into the prompt and tokenize the
  subtask target for the cross-entropy head.
"""
