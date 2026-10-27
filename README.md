# Affordable V2D Grounding

Robot grounding from human motion, object poses, and contact observations. The method has two parts: **primary dynamics correction** and **contact-orientation grounding**. A policy training pipeline turns the resulting episode artifacts into physically simulated robot rollouts and evaluation exports.

## Get started

- [Prepare motion and contact references](docs/correction/PIPELINE.md)
- [Run one episode or a batch of prepared episodes](docs/training/POLICY_HANDOFF.md)
- [Training method and the role of dynamics](docs/training/POLICY_METHOD.md)
- [CPU dependencies](docs/SETUP.md)

## Pipeline

```text
Human motion + object poses + contact observations
    -> robot retargeting and contact-orientation grounding
    -> dynamics correction and episode artifact export
    -> policy training and precision refinement
    -> full unassisted evaluation and trajectory export
```

Reference preparation preserves frame indices, coordinate transforms, observation masks, and source hashes. Tier 2 processing uses its supplied observations; clean references are used only for offline evaluation.

For a prepared episode, the policy pipeline runs acquisition, refinement, checkpoint evaluation, selection, and export automatically. Selection uses a fixed rule. Batch execution supports deterministic shards and resuming completed episodes. See the handoff for input requirements and commands.

## Repository layout

```text
correction/         Dynamics, contact geometry, reference correction, and export
retargeting/        Human-motion adapters and contact-aware G1/Dex3 kinematics
data_tools/         Dataset decoding and dependency asset downloads
training/
  scripts/         Pipeline, batch execution, preparation, and runtime setup
  core/            Policy training, refinement, and reference adapters
  evaluate/        Rollouts, leaderboard metrics, and official-format export
  configs/         Reusable configuration defaults
setup/              CPU environment specifications
docs/               Setup, method, and artifact interfaces
```

Datasets, model weights, generated results, private experiments, and tests are local development files and are excluded from the public source tree. Runtime dependencies and assets are downloaded separately; see [third-party notices](docs/THIRD_PARTY_NOTICES.md).
