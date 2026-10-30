# Affordable V2D Grounding

Robot grounding from human motion, object poses, and contact observations. The method has two parts: **primary dynamics correction** and **contact-orientation grounding**. A policy training pipeline turns the resulting episode artifacts into physically simulated robot rollouts and evaluation exports.

## Get started

From the installed native environment and this checkout, one command runs
an available episode through training, evaluation, selection and official export:

```bash
python training/scripts/run.py \
  --job /data/jobs/tier1/episode_000003 \
  --runtime /opt/v2d-runtime --output /runs/tier1/episode_000003 \
  --dataset-input /data/inputs/tier1 --submission-kit /opt/v2d_submission_kit
```

Wall-time limits and calendar deadlines are optional. `python -m training run` is
equivalent. The episode directory prefers its corrected job, otherwise uses its
original job with an explicit warning. Missing source samples are completed from
neighbouring observations with recorded repair masks; valid samples stay unchanged. Add
`--prepare-only` for CPU preflight. `--bundle` can prepare one episode from a
correction bundle first; `--jobs-root` runs prepared batches and shards.

The same source revision contains both tier-specific recipes, using 5M GPU replay:

| Input tier | Automatic training recipe | Checkpoint ranking |
| --- | --- | --- |
| Tier 1 | Native acquisition, then up to 90M precision transitions | MP-SR, SP-SR, ADD-AUC, lower MPPE |
| Tier 2/3 | Native acquisition, 60M precision, then up to 30M supported load transfer | ADD-AUC, lower MPPE, MP-SR, SP-SR |

The placement phase falls back explicitly to precision when its checked support
geometry or release window is inapplicable. Tier 3 uses the same configured
method but has not been GPU validated. These are training recipes, not a claim
that every episode succeeds. The entry point consumes prepared motion artifacts
and does not reconstruct raw video.

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

For a prepared episode, the policy pipeline runs acquisition, refinement,
checkpoint evaluation, selection, and export automatically. Selection uses the
fixed rule for its recipe. Batch execution supports deterministic shards and
resuming completed episodes. See the handoff for input requirements and commands.

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
