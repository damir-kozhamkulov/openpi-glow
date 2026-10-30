import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_resolve_train_episodes(monkeypatch):
    import pytest

    monkeypatch.delenv("OPENPI_TRAIN_EPISODES", raising=False)
    assert _data_loader.resolve_train_episodes() is None
    assert _data_loader.resolve_train_episodes("0:807") == list(range(807))  # the config's default
    monkeypatch.setenv("OPENPI_TRAIN_EPISODES", "0:379")
    assert _data_loader.resolve_train_episodes("0:807") == list(range(379))  # the environment wins
    assert _data_loader.resolve_train_episodes() == list(range(379))
    monkeypatch.setenv("OPENPI_TRAIN_EPISODES", "0,1,2")
    assert _data_loader.resolve_train_episodes() == [0, 1, 2]
    # Any subset: ranges and indices mixed, sorted, without duplicates.
    monkeypatch.setenv("OPENPI_TRAIN_EPISODES", "379:382")
    assert _data_loader.resolve_train_episodes() == [379, 380, 381]
    monkeypatch.setenv("OPENPI_TRAIN_EPISODES", "40, 2:4,3")
    assert _data_loader.resolve_train_episodes() == [2, 3, 40]
    for spec in ("5:5", "7:3", "-1", " , "):
        monkeypatch.setenv("OPENPI_TRAIN_EPISODES", spec)
        with pytest.raises(ValueError):
            _data_loader.resolve_train_episodes()


def test_episode_frame_indices():
    import torch

    table = {"from": torch.tensor([0, 3, 7, 12]), "to": torch.tensor([3, 7, 12, 14])}
    assert _data_loader.episode_frame_indices(table, [1, 3]) == [3, 4, 5, 6, 12, 13]
    assert _data_loader.episode_frame_indices(table, []) == []


def test_decision_rows():
    import torch

    table = {"from": torch.tensor([0, 3, 7, 12]), "to": torch.tensor([3, 7, 12, 14])}
    # Episode 1 (rows 3-6) has stages starting at frames 0 and 2; episode 3 (rows 12-13) is not
    # listed, so its rows appear once.
    rows = _data_loader.decision_rows(table, [1, 3], {1: (0, 2)}, window=1, repeat=3)
    assert rows == [3, 3, 3, 4, 5, 5, 5, 6, 12, 13]
    assert _data_loader.decision_rows(table, [1], {1: (0, 2)}, window=10, repeat=2) == [3, 3, 4, 4, 5, 5, 6, 6]
    assert _data_loader.decision_rows(table, [1, 3], {1: (0,)}, window=0, repeat=4) == [3, 4, 5, 6, 12, 13]
