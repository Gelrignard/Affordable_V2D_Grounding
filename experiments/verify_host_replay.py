"""GPU parity checks against native replay and the actual native SAC learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flash_chord.training.flash_sac import learner, replay
from flash_chord.training.flash_sac.checkpoint import (
    restore_learner_checkpoint,
    save_learner_checkpoint,
    training_checkpoint,
)
from flash_chord.training.flash_sac.config import (
    NetworkConfig,
    OptimizationConfig,
    ReplayConfig,
    TrainingConfig,
)

from training import host_replay


def equal(a, b, *, tolerance=0):
    left, right = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(left) == len(right)
    maximum = 0.0
    for x, y in zip(left, right, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape
        if tolerance:
            np.testing.assert_allclose(x, y, rtol=tolerance, atol=tolerance)
        else:
            np.testing.assert_array_equal(x, y)
        if x.size and x.dtype.kind == "f":
            assert np.isfinite(x).all() and np.isfinite(y).all()
            maximum = max(maximum, float(np.max(np.abs(x - y))))
    return maximum


def replay_case(n_step, dtype):
    spec = replay.ReplaySpec(
        capacity=37,
        minimum_size=8,
        batch_size=13,
        n_step=n_step,
        discount=0.99,
        world_count=4,
        observation_dim=9,
        action_dim=3,
        objective_term_count=4,
        critic_context_dim=3,
        observation_storage_dtype=dtype,
    )
    native = host_replay._native_create(spec)
    host = host_replay.create_host_replay(spec)
    rng = np.random.default_rng(876)
    checked = 0
    for step in range(65):
        if step in (23, 48):
            operation = replay.flush_pending if step == 23 else replay.clear_replay
            native, host = operation(native), operation(host)

        def random(width):
            return jnp.asarray(rng.normal(size=(4, width)).astype(np.float32))

        insert = replay.ReplayInsert(
            observation=random(9),
            action=random(3),
            objective_terms=random(4),
            terminated=jnp.asarray([(step + k) % 11 == 0 for k in range(4)]),
            truncated=jnp.asarray([(step + k) % 7 == 0 for k in range(4)]),
            next_observation=random(9),
            critic_context=random(3),
            next_critic_context=random(3),
        )
        native = replay.insert_replay(native, insert)
        host = replay.insert_replay(host, insert)
        equal(
            (native.size, native.write_index, native.pending),
            (host.size, host.write_index, host.pending),
        )
        size = int(native.size)
        if size:
            indices = jnp.arange(size, dtype=jnp.int32)
            equal(
                replay.gather_replay(native, indices),
                replay.gather_replay(host, indices),
            )
            key = jax.random.PRNGKey(step)
            a, b = replay.sample_replay(native, key), replay.sample_replay(host, key)
            equal(a, b)
            weights = jnp.asarray([1.0, 5.0, -0.1, 0.2], dtype=jnp.float32)
            equal(
                replay.relabel_rewards(a, weights, jnp.float32(0.02)),
                replay.relabel_rewards(b, weights, jnp.float32(0.02)),
            )
            checked += 1
    return {
        "n_step": n_step,
        "dtype": dtype,
        "checked_batches": checked,
        "exact_match": True,
        "host": host.storage.ring.snapshot(),
    }


def learner_case(out, mixed):
    config = TrainingConfig(
        world_count=8,
        total_environment_steps=800,
        updates_per_collection=4.0,
        mixed_precision=mixed,
        compute_dtype="float16" if mixed else "float32",
        replay=ReplayConfig(
            capacity=67,
            minimum_size=16,
            batch_size=16,
            n_step=7,
            observation_storage_dtype="float16",
        ),
        network=NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=16,
            critic_blocks=1,
            critic_hidden_dim=32,
            expansion=2,
            atom_count=11,
        ),
        optimization=OptimizationConfig(loss_scale_initial=128.0),
    )
    initial = jnp.arange(96, dtype=jnp.float32).reshape(8, 12) / 100
    context = jnp.tile(jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32), (8, 1))
    kwargs = {"action_dim": 3, "objective_term_count": 4}
    learner.create_replay = host_replay._native_create
    native, nx = learner.create_learner(config, initial, context, **kwargs)
    learner.create_replay = host_replay.create_host_replay
    host, hx = learner.create_learner(config, initial, context, **kwargs)
    maximum = 0.0
    rng = np.random.default_rng(987)
    for step in range(38):
        action = jnp.asarray(rng.uniform(-0.1, 0.1, (8, 3)).astype(np.float32))
        obs = jnp.asarray(rng.normal(size=(8, 12)).astype(np.float32))
        terminated = jnp.asarray(
            [(step + k) % 11 == 0 for k in range(8)], dtype=jnp.int32
        )
        truncated = jnp.asarray(
            [(step + k) % 13 == 0 for k in range(8)], dtype=jnp.int32
        )
        transition = learner.Transition(
            action=action,
            objective_terms=jnp.asarray(rng.uniform(0, 1, (8, 4)).astype(np.float32)),
            terminated=terminated,
            truncated=truncated,
            episode_reference_progress=jnp.ones(8, dtype=jnp.float32),
            termination_diagnostics=jnp.zeros((8, 0), dtype=jnp.float32),
            next_observation=obs,
            next_critic_context=context,
            post_reset_observation=obs + 0.1,
            post_reset_critic_context=context,
        )
        args = (
            transition,
            jnp.asarray([1.0, 5.0, 0.2, -0.1], dtype=jnp.float32),
            jnp.float32(0.02),
            jnp.float32(0.5),
            jnp.bool_(step == 22),
            jnp.bool_(True),
        )
        native, na, nm = nx.consume_transition(native, *args)
        host, ha, hm = hx.consume_transition(host, *args)
        # Compare every parameter, target, optimizer moment, clock, key, reward
        # normalizer and loss scale, plus actions and reported losses.
        maximum = max(
            maximum,
            equal(
                training_checkpoint(native), training_checkpoint(host), tolerance=2e-6
            ),
        )
        equal((na, nm), (ha, hm), tolerance=2e-6)
        assert (
            float(hm.actor_skipped_fraction) == float(hm.critic_skipped_fraction) == 0
        )
    paths = save_learner_checkpoint(out / ("mixed" if mixed else "float32"), host)
    fresh, _ = learner.create_learner(config, initial, context, **kwargs)
    restored, _ = restore_learner_checkpoint(paths.training_state, fresh)
    # Resume intentionally restores compact state, with empty replay and simulator reset.
    expected = training_checkpoint(host).replace(
        reward_normalizer=host.reward_normalizer.replace(
            discounted_return=jnp.zeros_like(host.reward_normalizer.discounted_return)
        )
    )
    equal(expected, training_checkpoint(restored))
    assert int(restored.replay.size) == 0
    return {
        "mixed_precision": mixed,
        "collections": 38,
        "optimizer_updates": int(host.global_update_step),
        "maximum_absolute_error": maximum,
        "checkpoint_restore_exact": True,
        "skipped_updates": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    assert jax.default_backend() == "gpu", "Must verify on the target GPU"
    host_replay.install()
    result = {
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "replay": [],
        "learner": [],
    }
    for n in (1, 7):
        for dtype in ("float16", "float32"):
            result["replay"].append(replay_case(n, dtype))
            print(json.dumps(result["replay"][-1]), flush=True)
    for mixed in (False, True):
        result["learner"].append(learner_case(out, mixed))
        print(json.dumps(result["learner"][-1]), flush=True)
    result.update(
        passed=True,
        elapsed_s=time.monotonic() - start,
        host_replay_sha256=hashlib.sha256(
            Path(host_replay.__file__).read_bytes()
        ).hexdigest(),
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
