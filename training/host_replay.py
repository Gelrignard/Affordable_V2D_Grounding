"""CPU-backed FIFO storage for the pinned native CHORD FlashSAC learner.

Only committed storage append/gather and allocation are replaced. Native JAX
n-step aggregation, terminal handling, random indices, reward relabeling and
all optimizer code stay in use. Ordered IO callbacks serialize mutable host
storage with the compiled learner; pure_callback would be incorrect here.
Replay contents are volatile, just as native checkpoints with save_replay=False.
"""

from __future__ import annotations

import dataclasses
import threading

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flash_chord.training.flash_sac import replay
from jax.experimental import io_callback

FIELDS = tuple(field.name for field in dataclasses.fields(replay.ReplayBatch))


class HostRing:
    """One actual transition history in ordinary system RAM, never a disk map."""

    def __init__(self, spec):
        self.spec = spec
        widths = {
            "observation": spec.observation_dim,
            "next_observation": spec.observation_dim,
            "action": spec.action_dim,
            "discounted_objective_terms": spec.objective_term_count,
            "critic_context": spec.critic_context_dim,
            "next_critic_context": spec.critic_context_dim,
        }
        self.arrays = {}
        for name in FIELDS:
            shape = (
                (spec.capacity, widths[name]) if name in widths else (spec.capacity,)
            )
            dtype = (
                np.dtype(spec.observation_storage_dtype)
                if name in ("observation", "next_observation")
                else np.dtype(np.float32 if name in widths else np.bool_)
            )
            self.arrays[name] = np.empty(shape, dtype=dtype)
        self.lock = threading.Lock()
        self.committed = 0
        self.read_batches = 0
        self.wraps = 0

    @property
    def nbytes(self):
        return sum(array.nbytes for array in self.arrays.values())

    def append(self, cursor, token, batch):
        index = int(np.asarray(cursor))
        count = self.spec.world_count
        if not 0 <= index < self.spec.capacity:
            raise ValueError("Invalid host replay write index")
        first = min(count, self.spec.capacity - index)
        with self.lock:
            for name in FIELDS:
                source = np.asarray(getattr(batch, name))
                target = self.arrays[name]
                if source.shape != (count, *target.shape[1:]):
                    raise ValueError(f"Unexpected host replay shape for {name}")
                # Same storage casts as the native device ring, notably float16.
                target[index : index + first] = source[:first]
                if first < count:
                    target[: count - first] = source[first:]
            self.committed += count
            self.wraps += int(index + count >= self.spec.capacity)
        return np.asarray(int(np.asarray(token)) + 1, dtype=np.int32)

    def gather(self, token, indices):
        del token  # Ordered effect and explicit token dependency protect append/read.
        indices = np.asarray(indices)
        if indices.dtype != np.int32 or indices.ndim != 1:
            raise ValueError(
                "Host replay indices must be a one-dimensional int32 array"
            )
        if np.any(indices < 0) or np.any(indices >= self.spec.capacity):
            raise ValueError("Host replay index out of range")
        with self.lock:
            result = {
                name: self.arrays[name][indices].astype(
                    np.bool_ if name in ("terminated", "truncated") else np.float32,
                    copy=False,
                )
                for name in FIELDS
            }
            self.read_batches += 1
        return replay.ReplayBatch(**result)

    def snapshot(self):
        with self.lock:
            return {
                "backend": "numpy_ram_ordered_jax_io_callback",
                "capacity": self.spec.capacity,
                "logical_bytes": self.nbytes,
                "committed_transitions_total": self.committed,
                "read_batches": self.read_batches,
                "wraps": self.wraps,
                "replay_saved_in_checkpoint": False,
            }


@flax.struct.dataclass
class HostStorage:
    token: jax.Array
    ring: HostRing = flax.struct.field(pytree_node=False)


_native_create = replay.create_replay
_native_append = replay._append_batch
_native_gather = replay._gather
_installed = False
_rings = []


def create_host_replay(spec, device=None):
    ring = HostRing(spec)
    # Allocate only the small native pending n-step window and a disposable
    # one-collection ring, never the requested capacity on the GPU.
    small_capacity = max(spec.world_count, spec.batch_size)
    small_spec = dataclasses.replace(
        spec,
        capacity=small_capacity,
        minimum_size=min(spec.minimum_size, small_capacity),
    )
    state = _native_create(small_spec, device=device)
    result = state.replace(
        spec=spec,
        storage=HostStorage(token=jnp.asarray(0, dtype=jnp.int32), ring=ring),
    )
    _rings.append(ring)
    return result


def _append(state, batch):
    if not isinstance(state.storage, HostStorage):
        return _native_append(state, batch)
    storage = state.storage
    token = io_callback(
        storage.ring.append,
        jax.ShapeDtypeStruct((), jnp.int32),
        state.write_index,
        storage.token,
        batch,
        ordered=True,
    )
    return state.replace(
        storage=storage.replace(token=token),
        size=jnp.minimum(state.size + state.spec.world_count, state.spec.capacity),
        write_index=(state.write_index + state.spec.world_count) % state.spec.capacity,
    )


def _gather(storage, indices):
    if not isinstance(storage, HostStorage):
        return _native_gather(storage, indices)
    signature = replay.ReplayBatch(
        **{
            name: jax.ShapeDtypeStruct(
                (indices.shape[0], *array.shape[1:]),
                np.bool_ if name in ("terminated", "truncated") else np.float32,
            )
            for name, array in storage.ring.arrays.items()
        }
    )
    return io_callback(
        storage.ring.gather, signature, storage.token, indices, ordered=True
    )


def install():
    """Opt in before native learner creation; do not modify the vendor checkout."""
    global _installed
    if _installed:
        return
    from flash_chord.training.flash_sac import learner

    replay._append_batch = _append
    replay._gather = _gather
    learner.create_replay = create_host_replay
    _installed = True


def snapshots():
    return [ring.snapshot() for ring in _rings]
