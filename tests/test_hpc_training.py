import copy
import os
import random
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from supermimic.networks import Critic
from test_policy import TinyTracker
from test_layer import LinearGeometry, NonlinearDynamics

from safety_mimic.ball import BallState
from safety_mimic.checkpoint import capture_rng, device_tree, load_snapshot, restore_rng, save_snapshot
from safety_mimic.distributed import Distribution
from safety_mimic.policy import BallConditionedPolicy
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.runtime import configure_runtime, state_digest
from safety_mimic.training import CriticBatch, actor_optimizer, clip_gradients, update_critic


def test_independent_actor_rates_cover_every_trainable_parameter_once():
    policy = BallConditionedPolicy(tracker=TinyTracker())
    optimizer = actor_optimizer(policy, 1e-5, 5e-5)
    assert [(g["name"], g["lr"]) for g in optimizer.param_groups] == [
        ("motion_encoder", 1e-5), ("action_decoder", 1e-5), ("ball_encoder", 5e-5),
    ]
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in policy.parameters() if p.requires_grad}


def test_snapshot_restores_optimizer_replay_and_every_rng(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    distribution = Distribution("cpu")
    torch.manual_seed(19)
    random.seed(20)
    np.random.seed(21)
    generator = torch.Generator().manual_seed(22)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        x = torch.randn(3, 2)
        target = torch.randn(3, 1, generator=generator)
        optimizer.zero_grad()
        (model(x) - target).square().mean().backward()
        optimizer.step()

    step()
    replay = [{"ball": BallState(torch.ones(1, 3), torch.zeros(1, 3), torch.ones(1)),
               "state": (torch.ones(1, 2),)}]
    path = save_snapshot(tmp_path, 1, {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                         {"replay": replay, "rng": capture_rng(generator, "cpu")}, distribution)
    expected_random = (random.random(), np.random.rand())
    step()
    expected = copy.deepcopy(model.state_dict())
    restored, local, _ = load_snapshot(tmp_path / "checkpoints/latest.json")
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    restore_rng(local["rng"], generator, "cpu")
    assert (random.random(), np.random.rand()) == expected_random
    assert isinstance(device_tree(local["replay"], "cpu")[0]["ball"], BallState)
    step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    # An unpublished partial generation cannot displace latest.json.
    (path.parent.parent / "partial_generation").mkdir()
    assert load_snapshot(tmp_path / "checkpoints/latest.json")[2] == path
    with pytest.raises(ValueError, match="same world size"):
        load_snapshot(path, world_size=2)
    with pytest.raises(RuntimeError, match="immutable"):
        save_snapshot(tmp_path, 1, {}, {}, distribution)
    (path.parent / "rank_00000.pt").unlink()
    with pytest.raises(ValueError, match="missing rank"):
        load_snapshot(path)


def distributed_worker(rank, port, directory):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    distribution = Distribution("cpu", timeout_seconds=60)
    try:
        model = torch.nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.1, -0.2]]))
        reference = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ref_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
        x = torch.arange(16, dtype=torch.double).reshape(8, 2) / 10
        target = torch.arange(8, dtype=torch.double)[:, None] / 7
        for indices in torch.arange(rank * 4, (rank + 1) * 4).split(2):
            ((model(x[indices]) - target[indices]).square().mean() / 2).backward()
        distribution.average_gradients(model.parameters())
        (reference(x) - target).square().mean().backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)
        clip_gradients(model.parameters())
        clip_gradients(reference.parameters())
        optimizer.step()
        ref_optimizer.step()
        torch.testing.assert_close(model.weight, reference.weight, atol=1e-12, rtol=1e-12)
        # Rank-local invalid gradients must reject the step on both ranks.
        model.weight.grad = torch.full_like(model.weight, float("inf") if rank else 1)
        before = model.weight.detach().clone()
        try:
            distribution.average_gradients(model.parameters())
            raise AssertionError("Invalid gradient accepted")
        except RuntimeError as failure:
            assert "Nonfinite local" in str(failure)
        torch.testing.assert_close(model.weight, before)
        # Exercise the same completed-gradient synchronization with actual
        # nested model Jacobians and implicitly differentiated QPs.
        actor = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            actor.weight.fill_(-0.7)
        reference_actor = copy.deepcopy(actor)
        layer = SafetyLayer(NonlinearDynamics(), LinearGeometry(), SafetyConfig(
            velocity_time=0, clearance_margin=0, trust_radius=2))
        ball = BallState(torch.zeros(1, 3, dtype=torch.double), torch.zeros(1, 3, dtype=torch.double),
                         torch.zeros(1, dtype=torch.double))
        def qp_loss(network, index):
            nominal = network(torch.ones(1, 1, dtype=torch.double))
            state = nominal.new_full((1, 1), 0.5 + index * 0.1)
            action = layer(state, ball, nominal, None, nominal * 0 - 2, nominal * 0 + 2).qp.action
            return action.square().mean()
        qp_loss(actor, rank).backward()
        distribution.average_gradients(actor.parameters())
        ((qp_loss(reference_actor, 0) + qp_loss(reference_actor, 1)) / 2).backward()
        torch.testing.assert_close(actor.weight.grad, reference_actor.weight.grad)
        torch.manual_seed(17)
        critic = Critic(3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        features = torch.randn(3, 2, 3) + rank
        batch = CriticBatch(features, torch.ones(2, 2), torch.zeros(2, 2))
        update_critic(critic, critic_optimizer, batch, epochs=2, minibatches=2, distribution=distribution)
        torch.testing.assert_close(critic.norm.mean, distribution.gather_tensor(features.flatten(0, 1)).mean(0))
        distribution.assert_identical({"critic": state_digest(critic), "actor": state_digest(model)})
        save_snapshot(directory, 1, {"model": model.state_dict()}, {"rank": rank}, distribution)
        common, local, _ = load_snapshot(os.path.join(directory, "checkpoints/latest.json"), rank=rank, world_size=2)
        assert common["world_size"] == 2 and local["rank"] == rank
    finally:
        distribution.close()


def test_two_process_sync_accumulation_critic_and_checkpoint(tmp_path):
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    mp.spawn(distributed_worker, args=(port, str(tmp_path)), nprocs=2, join=True)


def test_runtime_preserves_local_limits_and_partitions_slurm_cpus(monkeypatch):
    monkeypatch.delenv("SAFETYMIMIC_TASK_CPUS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assignments = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(20, 52)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: assignments.append(cpus))
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    for rank in range(2):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        assert configure_runtime(profile="slurm", cpu_threads=16) == f"cuda:{rank}"
    assert assignments == [set(range(20, 36)), set(range(36, 52))]
    with pytest.raises(ValueError, match="Local runs"):
        configure_runtime("cuda:1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="Local runs"):
        configure_runtime()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(32)))
    assert configure_runtime(cpu_threads=32) == "cuda:0"
    assert assignments[-1] == set(range(6))


def test_runtime_preserves_host_task_binding_and_thread_budget(monkeypatch):
    assignments = []
    threads = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(128)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: assignments.append(cpus))
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SAFETYMIMIC_TASK_CPUS", "20-23,30-33")
    for rank in range(2):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        configure_runtime(profile="slurm", cpu_threads=16)
    assert assignments == [set(range(20, 24)), set(range(30, 34))]
    assert threads == [4, 4]
