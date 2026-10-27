"""glow_module_diag: collapse statistics, attention grouping, and main() end to end.

The end-to-end tests need the LIBERO-10 + Goal plan pack, the local LeRobot copy (HF_LEROBOT_HOME
with repo `libero_staging`) and pi05_libero's norm_stats. They run dummy-size models with random
weights, so they check the wiring, not the numbers.
"""

import dataclasses
import json
import os
import pathlib

import numpy as np
import pytest

from . import glow_module_diag as gmd

# The dummy Gemma is 64 wide; give it a small SigLIP for the prefix unit tests.
from openpi.models_pytorch.glow_pi0_pytorch_test import tiny_vision_tower  # noqa: E402, F401
from openpi.glow import module_encoders as _enc  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]  # glow_repo (openpi is its submodule)
GOAL_PACK = REPO / "derived/plan_pack_libero10_goal.json"
DATASET = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "")) / "libero_staging"
ASSETS = REPO / "data/assets/pi05_libero"
NORM_STATS = ASSETS / "physical-intelligence/libero/norm_stats.json"

needs_data = pytest.mark.skipif(
    not (GOAL_PACK.exists() and (DATASET / "meta/info.json").exists() and NORM_STATS.exists()),
    reason="plan pack, local LeRobot copy or norm_stats not present",
)


def test_collapse_stats():
    same = gmd.collapse_stats(np.tile(np.arange(1.0, 5.0), (3, 1)))
    assert same["mean_cos"] == pytest.approx(1.0) and same["spread"] == pytest.approx(0.0, abs=1e-7)
    assert same["min_1_minus_cos"] == pytest.approx(0.0, abs=1e-7)
    orth = gmd.collapse_stats(np.eye(4) * 5.0)  # scale does not matter
    assert orth["mean_cos"] == pytest.approx(0.0) and orth["min_1_minus_cos"] == pytest.approx(1.0)
    assert orth["spread"] > 0.8


def test_attention_recorder_groups_and_masses():
    import torch

    # One sample: 4 image tokens (one padded), 2 module, 2 language, then 3 action tokens.
    prefix = torch.tensor([[0, 0, 0, -1, 1, 1, 2, 2]])
    t = prefix.shape[1] + 3
    rec = gmd.AttentionRecorder(num_layers=2)
    rec.prefix = prefix
    # Uniform attention over the 10 valid keys for every query; padding gets none.
    w = torch.zeros(1, 8, t, t)
    valid = torch.ones(t, dtype=torch.bool)
    valid[3] = False
    w[..., valid] = 1.0 / valid.sum()
    value = torch.ones(1, 1, t, 4)
    for _ in range(2):
        rec.record(w, value)
    s = rec.summary()
    assert s["keys_per_sample"] == {"images": 3, "module": 2, "language": 2, "actions": 3}
    a = s["by_query"]["actions"]
    assert a["mass_mean"]["module"] == pytest.approx(0.2) and a["mass_mean"]["images"] == pytest.approx(0.3)
    assert sum(a["mass_mean"].values()) == pytest.approx(1.0)
    # Uniform attention: every prefix group gets an average prefix token's share.
    for v in a["per_token_vs_prefix_mean"].values():
        assert v == pytest.approx(1.0)
    assert s["actions_value_weighted_share_mean"]["module"] == pytest.approx(0.2)
    # A second sample starts a new sequence (the next call is layer 0 again).
    rec.prefix = prefix
    rec.groups = None
    rec.record(w, value)
    assert rec.samples == 2 and rec.layer == 1


def _dummy_configs(monkeypatch):
    from transformers.models.auto import CONFIG_MAPPING

    import openpi.models_pytorch.gemma_pytorch as gemma_pytorch
    import openpi.shared.download as download
    import openpi.training.config as _config

    orig = CONFIG_MAPPING["paligemma"]

    def tiny():
        cfg = orig()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_hidden_layers = 2
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.patch_size = 56
        return cfg

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {**CONFIG_MAPPING, "paligemma": tiny})
    # Windows paths read as URL schemes ("w:"); the cluster never takes this path.
    real_download = download.maybe_download
    monkeypatch.setattr(
        download, "maybe_download", lambda p, **k: real_download(p, **k) if "://" in str(p) else pathlib.Path(p)
    )
    real_get = _config.get_config

    def get_config(name):
        c = real_get(name)
        small = {"paligemma_variant": "dummy", "action_expert_variant": "dummy", "pytorch_compile_mode": None}
        if hasattr(c.model, "module_width"):
            small["module_width"] = 32
        return dataclasses.replace(
            c,
            model=dataclasses.replace(c.model, **small),
            data=dataclasses.replace(c.data, plan_pack_path=str(GOAL_PACK)),
        )

    monkeypatch.setattr(_config, "get_config", get_config)
    return get_config


def _args(weights, tmp_path, **kw):
    return gmd.Args(
        weights_dir=str(weights),
        norm_stats_dir=str(ASSETS),
        dataset_root=str(DATASET),
        # LIBERO-10 (two-stage tasks) and Goal episodes, not a 0..n-1 prefix.
        episodes="0:6,379:385",
        num_samples=8,
        batch_size=4,
        device="cpu",
        out=str(tmp_path / "diag.json"),
        **kw,
    )


@needs_data
def test_main_module_model(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _dummy_configs(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(0)
    safetensors.torch.save_model(
        get_config("pi05_libero_glow_st").model.create_pytorch(), str(weights / "model.safetensors")
    )

    s = gmd.main(_args(weights, tmp_path))

    slots = s["collapse"]["slots"]
    model = get_config("pi05_libero_glow_st").model.create_pytorch()
    assert list(slots) == gmd.slot_names(model)
    pack_modules = len(s["collapse"]["modules"])
    assert slots["S0_OP"]["all"]["modules"] == pack_modules > 20
    assert slots["S0_OP"]["libero_goal"]["modules"] == 10  # every Goal task has one stage
    # The second group is padding for one-stage modules: only the two-stage ones count there.
    assert slots["S1_OP"]["libero_goal"]["modules"] == 0 and 0 < slots["S1_OP"]["all"]["modules"] < pack_modules
    # A random encoder separates modules; its CTX slot is a learned constant query, but the
    # transformer mixes the other slots into it.
    assert slots["S0_OBJ"]["all"]["spread"] > 0 and slots["S0_OBJ"]["all"]["mean_cos"] < 1.0

    o = s["flow"]["overall"]
    assert set(o) == set(gmd.FLOW_VARIANTS) and s["flow"]["samples"] == 8
    assert all(np.isfinite(v["flow"]) for v in o.values())
    for name in gmd.FLOW_VARIANTS:
        if name != "correct":
            assert o[name]["diff"] != 0.0, name
    # LIBERO-10 and Goal episodes are both in the sample.
    for name, by_suite in s["flow_by_suite"].items():
        assert set(by_suite) == {"all", "libero_10", "libero_goal"}, name
        assert by_suite["libero_10"]["samples"] + by_suite["libero_goal"]["samples"] == 8

    # Every variant's attention is recorded with its own layout.
    assert set(s["attention"]) == set(gmd.FLOW_VARIANTS)
    # Module keys per sample: none for the stock prefix; the model's own and the module-token
    # layouts mask the padding groups (7 tokens for a one-stage plan, 11 for a two-stage one);
    # the head-word layouts insert every token.
    n_all = _enc.SPATIAL_TOKENS + _enc.TEMPORAL_TOKENS
    for name, (_, layout) in gmd.FLOW_VARIANTS.items():
        a = s["attention"][name]
        keys = a["keys_per_sample"]["module"]
        if layout == "stock":
            assert keys == 0, name
        elif layout == "own" or layout[0] == "module":
            assert 7 <= keys <= 11, name
        else:
            assert keys == n_all, name
        assert sum(a["by_query"]["actions"]["mass_mean"].values()) == pytest.approx(1.0, abs=1e-4), name
    assert s["attention"]["plan_text"]["keys_per_sample"]["language"] > 2  # the plan text
    assert s["attention"]["task_prompt"]["keys_per_sample"]["language"] > 2  # the task string
    assert set(s["stage"]) == {"other_subgoal_vs_subgoal", "task_other_subgoal_vs_task_subgoal"}
    for pair in s["stage"].values():
        assert 2 <= pair["overall"]["samples"] < 8 and np.isfinite(pair["overall"]["diff"])

    att = s["attention"]["correct"]
    assert 7 <= att["keys_per_sample"]["module"] <= 11
    assert att["keys_per_sample"]["language"] == 2  # BOS and the newline answer token
    for q in ("actions", "images", "module"):
        assert sum(att["by_query"][q]["mass_mean"].values()) == pytest.approx(1.0, abs=1e-4)
    # Prefix queries never attend to the action tokens (stock prefix-LM mask).
    assert att["by_query"]["images"]["mass_mean"]["actions"] == pytest.approx(0.0, abs=1e-6)
    layers = get_config("pi05_libero_glow_st").model.create_pytorch().paligemma_with_expert.paligemma.config
    assert len(att["by_query"]["actions"]["mass_per_layer"]["module"]) == layers.text_config.num_hidden_layers
    assert sum(att["actions_value_weighted_share_mean"].values()) == pytest.approx(1.0, abs=1e-4)
    rms = s["embedding_rms"]
    assert set(rms["correct"]) == {"images", "module", "language"} and "module" not in rms["no_module"]
    # The encoder tokens are rescaled to the language tokens' RMS (pooled over the tokens; the
    # diagnostic reports the mean per-token RMS, so the two agree only closely); post_bos uses the
    # same scale, word_scale the prompt's word-token scale.
    assert rms["correct"]["module"] == pytest.approx(rms["correct"]["language"], rel=1e-2)
    assert rms["post_bos"]["module"] == pytest.approx(rms["correct"]["module"], rel=1e-5)
    assert rms["word_scale"]["module"] == pytest.approx(rms["post_bos_word_scale"]["module"], rel=1e-5)

    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    assert len(payload["per_sample_flow"]["correct"]) == 8


@needs_data
def test_main_init_loads_without_encoders(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    from openpi.models import pi0_config

    _dummy_configs(monkeypatch)
    weights = tmp_path / "init"
    weights.mkdir()
    stock = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pytorch_compile_mode=None,
    )
    torch.manual_seed(1)
    safetensors.torch.save_model(stock.create_pytorch(), str(weights / "model.safetensors"))

    with pytest.raises(Exception):  # noqa: B017 - strict load must refuse the missing encoders
        gmd.main(_args(weights, tmp_path))
    a = gmd.main(_args(weights, tmp_path, init=True))
    b = gmd.main(_args(weights, tmp_path, init=True))
    assert a["flow"]["overall"]["correct"]["flow"] == b["flow"]["overall"]["correct"]["flow"]  # seeded encoders
    # Without module tokens the init model is the stock one: both prompts' losses do not depend
    # on the encoders' seed.
    c = gmd.main(_args(weights, tmp_path, init=True, init_seed=5))
    # The variants that do not read the encoders do not depend on their seed.
    for name in ("no_module", "task_prompt", "plan_text", "head_pre_bos", "head_post_bos"):
        assert c["flow"]["overall"][name]["flow"] == a["flow"]["overall"][name]["flow"], name
    for name in ("correct", "post_bos", "word_scale", "post_bos_word_scale"):
        assert c["flow"]["overall"][name]["flow"] != a["flow"]["overall"][name]["flow"], name
    # --variants runs a subset (always with `correct`, the reference).
    d = gmd.main(_args(weights, tmp_path, init=True, variants=("no_module", "head_post_bos")))
    assert set(d["flow"]["overall"]) == {"correct", "no_module", "head_post_bos"}
    with pytest.raises(ValueError, match="unknown variants"):
        gmd.main(_args(weights, tmp_path, init=True, variants=("bogus",)))


@needs_data
def test_main_plan_prompt_reference(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _dummy_configs(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(0)
    safetensors.torch.save_model(
        get_config("pi05_libero_plan_prompt").model.create_pytorch(), str(weights / "model.safetensors")
    )

    s = gmd.main(_args(weights, tmp_path, config="pi05_libero_plan_prompt"))
    assert "collapse" not in s
    assert set(s["flow"]["overall"]) == {"correct"}
    att = s["attention"]["correct"]
    assert att["keys_per_sample"]["module"] == 0 and att["keys_per_sample"]["language"] > 2  # the plan text
    assert "module" not in att["by_query"]["actions"]["per_token_vs_prefix_mean"]
    assert sum(att["by_query"]["actions"]["mass_mean"].values()) == pytest.approx(1.0, abs=1e-4)


@needs_data
def test_main_anchored_init_is_the_text(monkeypatch, tmp_path):
    """A fresh text-anchored encoder emits its anchors whatever its seed: the --init flow loss and
    the tokens' collapse statistics do not depend on the encoders' initialization."""
    import safetensors.torch
    import torch

    from openpi.models import pi0_config

    _dummy_configs(monkeypatch)
    weights = tmp_path / "init"
    weights.mkdir()
    stock = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pytorch_compile_mode=None,
    )
    torch.manual_seed(1)
    safetensors.torch.save_model(stock.create_pytorch(), str(weights / "model.safetensors"))

    runs = [
        gmd.main(_args(weights, tmp_path, config="pi05_libero_glow_st_anchor", init=True, init_seed=seed))
        for seed in (0, 5)
    ]
    a, b = (r["flow"]["overall"]["correct"]["flow"] for r in runs)
    assert a == pytest.approx(b, rel=1e-6)
    for slot in runs[0]["collapse"]["slots"]:
        sa, sb = (r["collapse"]["slots"][slot]["all"] for r in runs)
        if sa["modules"] < 2:
            continue  # a padding group for every module
        assert sa["mean_cos"] == pytest.approx(sb["mean_cos"], abs=1e-6), slot


def _prefix_case(**model_kw):
    """A dummy-size module model, an observation and its preprocessed prefix inputs."""
    from openpi.models_pytorch.glow_module_pi0_pytorch_test import _model_with
    from openpi.models_pytorch.glow_module_pi0_pytorch_test import _module_arrays
    from openpi.models_pytorch.glow_module_pi0_pytorch_test import _obs

    m = _model_with(**model_kw)
    obs = _obs(0, _module_arrays(0))
    images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(obs, train=False)  # noqa: SLF001
    return m, obs, (images, img_masks, lang_tokens, lang_masks)


def test_make_prefix_reproduces_the_model_prefix():
    import torch

    m, obs, inputs = _prefix_case()
    want = m.embed_observation_prefix(obs, *inputs)
    n_lang, n_mod = inputs[2].shape[1], len(gmd.slot_names(m))
    n_img = want[0].shape[1] - n_lang - n_mod
    valid_module = m.module_token_mask(obs)
    assert not valid_module.all() and valid_module.any()  # two-stage plans: two groups are padding
    for kw in (
        {"own": n_mod},
        {"tokens": m.module_tokens, "token_mask": m.module_token_mask, "where": "pre_bos", "scale": "language"},
    ):
        rec = gmd.AttentionRecorder(1)
        prefix = gmd.make_prefix(m, rec, **kw)
        got = prefix(obs, *inputs)
        for x, y in zip(got, want, strict=True):
            torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)
        groups = rec.prefix
        module = groups[:, n_img : n_img + n_mod]
        assert (module[valid_module] == gmd.GROUPS.index("module")).all() and (module[~valid_module] == -1).all(), kw
        valid_lang = groups[:, n_img + n_mod :][want[1][:, n_img + n_mod :]]
        assert (valid_lang == gmd.GROUPS.index("language")).all() and (groups[~want[1]] == -1).all()
        assert set(prefix.rms) == {"images", "module", "language"}
    # Without tokens it is the stock prefix.
    rec = gmd.AttentionRecorder(1)
    stock = gmd.make_prefix(m, rec)(obs, *inputs)
    for x, y in zip(stock, m.embed_prefix(*inputs), strict=True):
        assert torch.equal(x, y)
    assert not (rec.prefix == gmd.GROUPS.index("module")).any()


def test_make_prefix_word_scale_is_the_words_model():
    """The word_scale variant at the module-text RMS is what a module_token_scale="words" model
    builds itself."""
    import torch

    m, obs, inputs = _prefix_case(module_token_scale="words")
    want = m.embed_observation_prefix(obs, *inputs)
    target = m.module_text_rms(obs)[:, 0, 0]
    got = gmd.make_prefix(
        m,
        gmd.AttentionRecorder(1),
        tokens=m.module_tokens,
        token_mask=m.module_token_mask,
        scale="word",
        target_rms=target,
    )(obs, *inputs)
    for x, y in zip(got, want, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)


def test_make_prefix_after_bos_and_scales():
    import torch

    m, obs, inputs = _prefix_case()
    n_lang, n_mod = inputs[2].shape[1], len(gmd.slot_names(m))
    stock = m.embed_prefix(*inputs)
    n_img = stock[0].shape[1] - n_lang
    rec = gmd.AttentionRecorder(1)
    pre = gmd.make_prefix(m, rec, tokens=m.module_tokens, where="pre_bos", scale="language")(obs, *inputs)
    post = gmd.make_prefix(m, rec, tokens=m.module_tokens, where="post_bos", scale="language")(obs, *inputs)
    # The first language token stays right after the images; the module tokens follow it. Without
    # a token mask every inserted token is valid.
    torch.testing.assert_close(post[0][:, : n_img + 1], stock[0][:, : n_img + 1])
    torch.testing.assert_close(post[0][:, n_img + 1 + n_mod :], stock[0][:, n_img + 1 :])
    torch.testing.assert_close(post[0][:, n_img + 1 : n_img + 1 + n_mod], pre[0][:, n_img : n_img + n_mod])
    assert post[1][:, n_img + 1 : n_img + 1 + n_mod].all() and not post[2][:, n_img + 1 : n_img + 1 + n_mod].any()
    assert (rec.prefix[:, n_img + 1 : n_img + 1 + n_mod] == gmd.GROUPS.index("module")).all()
    assert (rec.prefix[:, n_img] == gmd.GROUPS.index("language")).all()

    target = torch.arange(1, inputs[2].shape[0] + 1, dtype=torch.float32) * 2.0
    word = gmd.make_prefix(m, rec, tokens=m.module_tokens, scale="word", target_rms=target)(obs, *inputs)
    valid = m.module_token_mask(obs)
    module_rms = word[0][:, n_img : n_img + n_mod][valid].pow(2).mean(-1).sqrt()
    expected = target[:, None].expand(-1, n_mod)[valid]
    torch.testing.assert_close(module_rms, expected, rtol=1e-4, atol=1e-4)

    fixed = torch.randn(inputs[2].shape[0], 3, stock[0].shape[-1])
    raw = gmd.make_prefix(m, rec, tokens=lambda _obs: fixed, scale="raw")(obs, *inputs)
    torch.testing.assert_close(raw[0][:, n_img : n_img + 3], fixed)
    with pytest.raises(ValueError, match="placement"):
        gmd.make_prefix(m, rec, tokens=m.module_tokens, where="middle")(obs, *inputs)
    with pytest.raises(ValueError, match="scale"):
        gmd.make_prefix(m, rec, tokens=m.module_tokens, scale="huge")(obs, *inputs)


def test_word_rms_skips_bos_and_newline():
    import torch

    embed = torch.nn.Embedding(10, 4)
    tokens = np.array([[2, 5, 6, 7, 0], [2, 4, 7, 0, 0]])
    mask = np.array([[True, True, True, True, False], [True, True, True, False, False]])
    got = gmd.word_rms(embed, tokens, mask, 3.0, "cpu")
    w = embed.weight.detach() * 3.0
    rms = w.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(got, torch.stack([(rms[5] + rms[6]) / 2, rms[4]]))


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="Goal plan pack not present")
def test_head_words():
    from openpi.glow import plan_pack as _plan_pack

    pack = _plan_pack.PlanPack.load(GOAL_PACK)
    by_index = {t.task_index: t for t in pack.tasks.values()}
    group = ["ATOM", "OP", "OBJ", "TGT"]
    slots = [f"S{g}_{s}" for g in range(4) for s in group] + ["CTX", *gmd.TEMPORAL_SLOTS]
    mugs = by_index[0]  # put the white mug on the left plate, then the yellow and white mug on the right
    words = gmd.head_words(mugs, 0, slots)
    assert words[:8] == ["on", "put", "mug", "plate", "on", "put", "mug", "plate"]
    assert words[8:16] == [None] * 8 and words[16:] == ["on", "two", "first"]
    assert gmd.head_words(mugs, 1, slots)[-1] == "second"
    # The slot order places stages: order [1, 0] shows stage 1 in group 0.
    microwave = by_index[2]  # the mug into "microwave (heating region)", then close it (no target)
    assert gmd.head_words(microwave, 0, slots)[2] == "mug" and gmd.head_words(microwave, 0, slots)[3] == "microwave"
    assert gmd.head_words(microwave, 1, slots)[4:8] == ["close", "close", "microwave", None]
    assert gmd.head_words(microwave, 1, slots, order=[1, 0, 2, 3])[:4] == ["close", "close", "microwave", None]
    assert gmd.head_words(microwave, 1, slots, order=[1, 0, 2, 3])[16] == "close"
    drawer = by_index[12]  # open the top drawer and put the bowl inside: one stage
    assert gmd.head_words(drawer, 0, slots)[:4] == ["in", "open", "bowl", "drawer"]
    assert gmd.head_words(drawer, 0, slots)[4:16] == [None] * 12
    assert gmd.head_words(drawer, 0, slots)[16:] == ["in", "one", "first"]
    assert gmd.head_words(drawer, 0, ["S0_OBJ", "T_ACTIVE"]) == ["bowl", "first"]


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="Goal plan pack not present")
def test_stage_prompts():
    from openpi.glow import plan_pack as _plan_pack

    pack = _plan_pack.PlanPack.load(GOAL_PACK)
    by_index = {t.task_index: t for t in pack.tasks.values()}
    mugs = by_index[0]
    p = gmd.stage_prompts(mugs, 1)
    assert set(p) == set(gmd.STAGE_PROMPTS)
    assert p["subgoal"] == "put the yellow and white mug on the right plate"
    assert p["other_subgoal"] == "put the white mug on the left plate"
    assert p["task_subgoal"] == f"{mugs.task}. {p['subgoal']}"
    assert p["task_other_subgoal"] == f"{mugs.task}. {p['other_subgoal']}"
    one = by_index[12]
    q = gmd.stage_prompts(one, 0)
    assert q["subgoal"] == q["other_subgoal"] and q["task_subgoal"] == q["task_other_subgoal"]
    # Every task string + subgoal fits the stage prompt without truncation.
    from openpi.models import tokenizer as _tokenizer

    tok = _tokenizer.PaligemmaTokenizer(gmd.STAGE_PROMPT_LEN)
    for t in pack.tasks.values():
        for k in range(t.num_plans):
            for text in gmd.stage_prompts(t, k).values():
                assert not tok.tokenize(text)[1][-1], text


def test_stage_summary():
    tasks = np.array([0, 0, 1, 1, 12])
    stages = np.array([0, 1, 0, 1, 0])
    multistage = np.array([True, True, True, True, False])
    flows = {
        "subgoal": np.array([1.0, 1.0, 2.0, 2.0, 9.0]),
        "other_subgoal": np.array([1.5, 1.5, 2.0, 3.0, 0.0]),
    }
    s = gmd.stage_summary(flows, tasks, stages, multistage)
    assert list(s) == ["other_subgoal_vs_subgoal"]  # the task_* pair is absent
    o = s["other_subgoal_vs_subgoal"]
    assert o["overall"]["samples"] == 4 and o["overall"]["diff"] == pytest.approx(0.5)
    assert o["per_task"][0]["diff"] == pytest.approx(0.5) and o["per_task"][1]["diff"] == pytest.approx(0.5)
    assert set(o["per_task"]) == {0, 1}  # the one-stage task is left out
    assert o["by_stage"][0]["diff"] == pytest.approx(0.25) and o["by_stage"][1]["diff"] == pytest.approx(0.75)
