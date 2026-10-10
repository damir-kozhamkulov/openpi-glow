"""Fully sharded data parallel (FSDP2) training support for PI0Pytorch / GlowPI0Pytorch.

A float32 pi05 replica with AdamW needs ~58 GB per GPU under DDP (parameters 14.5 + gradients
14.5 + moments 29), which no 46-48 GB card holds. Sharding gradients, moments and the resting
parameters across N GPUs brings that to 58 / N, plus one full copy of the parameters while a
forward/backward is in flight.

Sharding units. `PaliGemmaWithExpertModel.forward` runs each joint layer by calling the
submodules of `layers[i]` directly (input_layernorm, q/k/v/o_proj, post_attention_layernorm,
mlp), never `GemmaDecoderLayer.forward`, so FSDP hooks on the decoder layer would never fire.
Each decoder layer is therefore sharded as a group of exactly those submodules, which both the
joint loop and the stock HF decoder-layer forward (the GLOW prefix-only CE pass) call. SigLIP
encoder layers are called normally and are sharded whole; everything else (embeddings, the
tied lm_head, projections, final norms) belongs to the root unit.

Parameters stay unsharded from the forward that first uses them until `reshard` after the
backward: one all-gather and one reduce-scatter per micro-batch, the communication volume of a
DDP allreduce. Resharding between forward and backward is not safe for this model: the joint
transformer runs inside nested activation checkpoints whose recomputation re-enters the
layers' forward hooks, and the last PaliGemma layer's attention output and MLP receive no
gradient in the flow loss, so FSDP's pre-backward unshard never fires for it while its k/v
projections still need their weights. The cost is a full parameter copy (~14.5 GB float32)
during each forward/backward.

Checkpoints keep the single-process format: `model.safetensors` is byte-for-byte what
`safetensors.torch.save_model` writes for the unwrapped model, and `optimizer.pt` is a plain
`torch.optim.Optimizer.state_dict()` with full tensors. A run can therefore resume under FSDP
from a DDP checkpoint and the other way round, and evaluation reads FSDP checkpoints unchanged.
"""

import os

import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

PARALLEL_MODES = ("ddp", "fsdp")

# The submodules of a Gemma decoder layer that the joint layer loop calls one by one.
_DECODER_LAYER_PARTS = ("input_layernorm", "post_attention_layernorm", "mlp")
_ATTENTION_PARTS = ("q_proj", "k_proj", "v_proj", "o_proj")


def resolve_parallelism() -> str:
    """Multi-GPU strategy, from `OPENPI_PARALLEL` (default "ddp")."""
    mode = os.environ.get("OPENPI_PARALLEL", "ddp").strip().lower()
    if mode not in PARALLEL_MODES:
        raise ValueError(f"OPENPI_PARALLEL must be one of {PARALLEL_MODES}, got {mode!r}")
    return mode


def build_mesh(device: torch.device) -> DeviceMesh:
    """A 1-D mesh over the default process group (no new communicators)."""
    return DeviceMesh.from_group(dist.group.WORLD, device.type)


def _decoder_layer_group(layer: torch.nn.Module) -> list[torch.nn.Module]:
    return [getattr(layer, name) for name in _DECODER_LAYER_PARTS] + [
        getattr(layer.self_attn, name) for name in _ATTENTION_PARTS
    ]


def sharding_units(model: torch.nn.Module) -> list[list[torch.nn.Module]]:
    """The module groups sharded below the root, in forward order."""
    pwe = model.paligemma_with_expert
    units = [[layer] for layer in pwe.paligemma.vision_tower.vision_model.encoder.layers]
    for lm_layer, expert_layer in zip(pwe.paligemma.language_model.layers, pwe.gemma_expert.model.layers, strict=True):
        units.append(_decoder_layer_group(lm_layer))
        units.append(_decoder_layer_group(expert_layer))
    return units


def shard_model(model: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
    """Shards `model` in place (FSDP2) and returns it. Load weights before calling this."""
    for unit in sharding_units(model):
        fully_shard(unit, mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    model.set_reshard_after_backward(False)
    return model


def reshard(model: torch.nn.Module) -> None:
    """Frees the unsharded parameters after a backward.

    Required before reading gradients or stepping: while unsharded, `model.parameters()` yields
    the full-size working copies, whose gradients were already moved into the shards.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def clip_grad_norm_(parameters, max_norm: float) -> torch.Tensor:
    """`torch.nn.utils.clip_grad_norm_` over sharded gradients: the total norm is reduced
    across ranks before the clip coefficient is applied."""
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads)
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm)
    return total_norm


def _gather(value, *, is_main: bool):
    """The full tensor on the main rank (None elsewhere). Collective for DTensors."""
    full = value.full_tensor() if isinstance(value, DTensor) else value
    return full.detach().cpu() if is_main else None


def full_model_state_dict(model: torch.nn.Module, *, is_main: bool) -> dict[str, torch.Tensor]:
    """The unsharded `state_dict()` on CPU on the main rank, {} elsewhere. Collective.

    Tied parameters (the PaliGemma embedding and lm_head) are gathered once and appear under
    every name as the same tensor, as in an unwrapped model's state_dict, so
    `save_model_file` drops the same duplicate names `save_model` would.
    """
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    gathered: dict[int, torch.Tensor | None] = {}
    state = {}
    for name in model.state_dict():
        obj = tensors[name]
        if id(obj) not in gathered:
            gathered[id(obj)] = _gather(obj, is_main=is_main)
        if is_main:
            state[name] = gathered[id(obj)]
    return state


def save_model_file(state_dict: dict[str, torch.Tensor], path) -> None:
    """`safetensors.torch.save_model` for a state dict instead of a module."""
    state_dict = dict(state_dict)
    metadata = None
    for kept_name, removed_names in safetensors.torch._remove_duplicate_names(state_dict).items():  # noqa: SLF001
        for name in removed_names:
            metadata = metadata or {}
            metadata[name] = kept_name
            del state_dict[name]
    safetensors.torch.save_file({k: v.contiguous() for k, v in state_dict.items()}, path, metadata=metadata)


def full_optimizer_state_dict(optim: torch.optim.Optimizer, *, is_main: bool) -> dict:
    """`optim.state_dict()` with every sharded state tensor gathered, on the main rank ({} elsewhere).
    Collective.

    The integer parameter indices are the optimizer's parameter order, which is
    `model.parameters()` order with or without FSDP, so the result is interchangeable with a
    DDP or single-GPU run's optimizer.pt.
    """
    sharded = optim.state_dict()
    state = {}
    for index, entry in sharded["state"].items():
        full_entry = {
            key: _gather(value, is_main=is_main) if torch.is_tensor(value) else value for key, value in entry.items()
        }
        if is_main:
            state[index] = full_entry
    return {"state": state, "param_groups": sharded["param_groups"]} if is_main else {}


def _local_slice(full: torch.Tensor, param: DTensor) -> torch.Tensor:
    """This rank's shard of `full`, a tensor shaped like the parameter, under the parameter's
    Shard(0) placement (torch.chunk boundaries)."""
    rank = param.device_mesh.get_local_rank()
    world = param.device_mesh.size()
    rows = full.shape[0]
    chunk = -(-rows // world)
    start = min(rank * chunk, rows)
    local = full[start : min(start + chunk, rows)]
    expected = param.to_local().shape
    if local.shape != expected:
        raise RuntimeError(
            f"shard of {tuple(full.shape)} is {tuple(local.shape)}, parameter shard is {tuple(expected)}"
        )
    return local


def load_optimizer_state_dict(optim: torch.optim.Optimizer, path) -> None:
    """Loads a full (single-process format) optimizer.pt into an optimizer over sharded parameters.

    Every rank memory-maps the file and copies only its own rows of each state tensor, so no rank
    holds the full ~29 GB moments.
    """
    full = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    params = [p for group in optim.param_groups for p in group["params"]]
    if len(full["param_groups"]) != len(optim.param_groups) or sum(
        len(g["params"]) for g in full["param_groups"]
    ) != len(params):
        raise ValueError(f"{path}: parameter layout does not match this optimizer ({len(params)} parameters)")
    state = {}
    for index, entry in full["state"].items():
        param = params[index]
        sharded_entry = {}
        for key, value in entry.items():
            if torch.is_tensor(value) and isinstance(param, DTensor) and value.shape == param.shape:
                value = DTensor.from_local(  # noqa: PLW2901
                    _local_slice(value, param).to(param.device, copy=True),
                    param.device_mesh,
                    param.placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                )
            sharded_entry[key] = value
        state[index] = sharded_entry
    optim.load_state_dict({"state": state, "param_groups": full["param_groups"]})
