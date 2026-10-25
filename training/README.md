# Policy training after trajectory correction

This workstream implements one residual manipulation policy per episode using
the Tier 1/2 handoff. Tier 3 reconstruction remains separate. Full-body physics,
G1/Dex3 assets and frozen SONIC come from NVIDIA's pinned FlashCHORD runtime.
Our learner is PyTorch; reproducing the full CHORD training recipe is not required.

**Status:** CPU learning tests, NVIDIA reference-loader integration on all 24
prepared jobs, and submission-kit trajectory-reader integration pass. CUDA
simulation, the container build, policy quality and 4090 performance remain to
be checked on the cloud machine. No G1 policy has been trained yet.

## Implemented learning pipeline

```text
same-tier pose reference + object goal + contact regions
                         |
          residual actor (arms, waist and fingers)
                         |
        frozen SONIC body controller + finger targets
                         |
           full G1/Dex3 + object + support physics
                         |
    actual states/actions/contacts/rewards, including failures
                         |
        SAC updates + optional learned-model updates
                         |
  short differentiable imagined rollouts -> additional actor updates
```

- `sac`: a compact real-transition SAC control experiment.
- `model_based`: the same SAC learner plus a small closed-loop world model,
  short backpropagated imagined returns and a terminal critic. This is our
  adaptation of model-based policy learning, not an exact SuperMimic reproduction
  or a pretrained foundation world model.
- The residual actor is the learned policy. There is no separate task-specific
  feedback-policy training stage or cloning of infeasible reference actions.

The world model learns **SONIC + residual wrapper + physics** from executed
transitions. Its state includes robot position/velocity, object pose/velocity,
live contacts, SONIC's ordered 10-step history, previous controller actions,
target-delay history, and episode context. Its goal includes reference poses,
the frozen SONIC encoder's future-reference latent, contact capability, object
geometry/mass/inertia, and support geometry. The actor receives native policy
observations, explicit live contacts and the goal; the critic/model also receive
the extra controller state.

Model fitting uses contiguous windows with masks at episode boundaries. Separate
world IDs supply held-out validation data. Imagined updates start only after the
minimum model-update count and validation gates pass; model weights are frozen
during these actor updates. Actual simulation and model fitting continue during
training. The gates are initial hyperparameters, not evidence of sample savings.
Imagined actions are regularized toward executed replay actions to limit model
extrapolation. Contact-force features use 100 N units; physical traces retain N.

## Reference and controller contract

See [the correction handoff](../docs/correction/RL_HANDOFF_2026-10-24.md) and
[paired pilot](../docs/correction/PAIRED_PILOT_RUN_2026-10-24.md). The roster is
episodes **3, 5, 11, 16, 18 and 25**, in both tiers.

References are soft priors. Hashes, tier, frame IDs, timestamps, observation
masks, coordinate transforms and physical-validation labels survive preparation.
A failed validation label does not prevent soft-reference use and is never
changed into an expert-demonstration label. The saved 30 Hz diagnostic rollouts
are not world-model transition labels; new transitions come from the simulator.

The action order is NVIDIA's 29 SONIC body residuals plus 14 Dex3 finger residuals.
A tanh-bounded actor produces normalized actions, multiplied by `action_bound`
(default 3) before NVIDIA's wrapper. Its body/finger residual scales remain 0.15.
These are joint-target residuals, not torque or wrench commands. Hip, knee and
ankle residuals are zero by default; SONIC still controls those joints. Set
`lower_body_residual` in [0,1] to enable bounded corrections. Full-body collisions
and fall termination remain active.

Reward combines NVIDIA's balance, joint-limit, smoothness, progress and
object-position terms with weak posture/palm tracking, object orientation, and
a CHORD-inspired contact-wrench capability descriptor. The descriptor compares
friction-cone support from **live robot-on-object contacts** with the soft contact
target on a common deterministic basis. It is geometric guidance, not proof of
torque-limited dynamic feasibility. Missing measured contacts earn no capability
bonus. Planned `force_ff` remains in the input bundle but is neither applied to
the object nor treated as measured force or action labels.

Missing observations receive reduced confidence. Contact confidence also reflects
the handoff's static/wrench checks, without treating them as dynamic certification.
VOC object assistance, reset assistance and reset freeze are disabled. Explicit
reference-phase resets are training initializations; 25% start at frame zero.

`original` means **our contact-aware retargeting before correction**; `corrected`
means the exported corrected candidate. Both deliberately share contact guidance
to compare the robot pose reference under the same downstream setup. Neither is
NVIDIA's body-only retargeting baseline. That third comparison requires a separate
NVIDIA-retargeting reference; do not label `original` as `nvidia_body_baseline`.

## Prepare portable jobs

Use Python 3.11 or 3.12 from this worktree's repository root. CPU dependencies are
in `training/requirements-cpu.txt`; the primary checkout's existing `.soma-env`
can run local preparation/tests.

```bash
python -m training inspect --bundle /path/to/rl_pilot_handoff_20261024
python -m training prepare \
  --bundle /path/to/rl_pilot_handoff_20261024 \
  --urdf /path/to/g1/main_with_hand.urdf \
  --output outputs/training_jobs --all --variants original corrected
```

Use `--tier 2 --episode 18` instead of `--all` for one episode. Preparation checks
named-joint FK against supplied palm poses and writes `motion_v1` Parquet,
reference/guidance NPZs, geometry, assets and `job.json` hashes. Existing nonempty
output directories are rejected. All **24 prepared jobs** are already available
locally at `outputs/training_jobs_v1/` in this worktree. Generated data is ignored
by Git and must be copied separately to the cloud.

The object URDF follows the **pinned NVIDIA `soma_to_g1.py` template**: mass 1 kg,
zero local COM and diagonal inertia 0.01 kg m². These are baseline defaults, not
identified object properties. Physics uses NVIDIA's scene builder, friction 1,
object free-joint damping 0.01 and convex decomposition. Meshes retain their local
frame. Empty support USDA files mean ground-only scenes. The simulator's actual
mass/inertia are recorded and included in model context.

Jobs can be relocated: the runtime creates a temporary Parquet with rebased asset
paths without altering hashed inputs. Windows CRLF and Linux LF versions of the
known G1 URDF are checked against one canonical robot hash.

## Cloud runtime setup

Runtime pin: `nvidia-isaac/video_to_data` at
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. The CPU retargeting checkout under
`upstream/` stays separate. Setup does not start training.

The optional container excludes NVIDIA's JAX/FlashSAC training extras:

```bash
docker build -f training/Dockerfile -t v2d-policy:pilot training
docker run --rm --gpus all \
  -v /absolute/path/to/jobs:/jobs:ro \
  -v /absolute/path/to/runs:/runs \
  v2d-policy:pilot python -m training doctor --runtime /opt/v2d-runtime
```

The recipe follows the pinned runtime's driver assumption (host driver >=580).
It has not been built locally. Alternatively, in a dedicated Linux environment:

```bash
python -m training.bootstrap --runtime vendor/flash-runtime --download-assets
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vendor/flash-runtime/robotic_grounding/flash_chord
python -m training doctor --runtime vendor/flash-runtime
```

Bootstrap requires Git/Git LFS and fetches G1 meshes and SONIC weights. It refuses
to alter an existing unpinned or dirty runtime. `doctor` reports dependencies,
CUDA and asset hashes; environment construction verifies SONIC hashes and the
clean runtime commit. A 133-byte LFS pointer is not a model. `doctor` is a readiness
report, not a completed simulator smoke test.

## Profile, train and evaluate

The 4090 config starts with 256 worlds and CPU replay, keeping the augmented
state replay off the 24 GB GPU. Profile before choosing final world/batch counts.

```bash
python -m training profile \
  --job /jobs/tier2/episode_000018/corrected \
  --runtime /opt/v2d-runtime --config training/configs/4090_pilot.json \
  --output /runs/profile --hours 0.1 --transitions 250000

python -m training train \
  --job /jobs/tier2/episode_000018/corrected \
  --runtime /opt/v2d-runtime --config training/configs/4090_pilot.json \
  --output /runs/episode18 --algorithm model_based --hours 2

python -m training evaluate \
  --job /jobs/tier2/episode_000018/corrected \
  --runtime /opt/v2d-runtime --checkpoint /runs/episode18/latest.pt \
  --output /runs/episode18_eval --worlds 1
```

Use `--algorithm sac` under identical settings for the learner comparison.
CLI overrides include `--worlds`, `--device`, `--seed`, `--hours`, `--transitions`.
Training stops at the first budget reached. Counts can overshoot by less than
one vector batch. The wall budget covers the training loop, excluding environment
initialization, shutdown checkpoint I/O and evaluation.

`metrics.jsonl` records transitions, real/model updates, validation, imagination
status, elapsed time, throughput, replay size and GPU memory. PyTorch peak memory
does not include all Warp/ONNX allocations, so device-wide used memory is also
sampled. Use a steady-state profile window for estimates. Default 2 hours / 10M
transitions is an experiment cap, not a quality or 90-episode cost prediction.

Outputs: `latest.pt`, periodic checkpoints, `run.json`, `report.json`, and a
`world_model.pt` when model updates occurred. No checkpoint is named “best” without
evaluation. Evaluate periodic checkpoints to obtain quality-versus-wall-time
curves; low model loss is not a task-success measure.

Resume into a **new output directory** with `--resume /old/run/latest.pt`. Weights,
optimizers and counters resume, but replay is fresh; this is not an exact
stochastic continuation. New budgets are additional. `--warm-model` transfers
only the model, with a fresh per-episode actor/critic. Transfer rejects different
tiers, reference variants and controller/state/reward schemas, and revalidates
against fresh data.

Evaluation starts at frame zero, performs no within-episode resets or assistance,
and continues after tracking/fall failures. It records initial and actual 50 Hz
states, then interpolates physical recordings back to **all source timestamps**.
No time warping or invisible-frame removal is performed. Outputs are
`control_trace.npz`, `evaluation.json`, and `object_trajectories.parquet` using the
submission kit's xyzw/metre convention. Invalid worlds remain as zero placeholders
with `environment_valid=false`; the raw trace retains diagnostic nonfinite data.

The Parquet is tested against `v2dlb.submission_parquet.read_trajectory`. This is
our custom PyTorch evaluator, with its own code/checkpoint hashes. NVIDIA's
FlashSAC loader does not load these `.pt` files. Official packing/scoring and
custom-policy integration verification remain a cloud/submission step. Local
errors use same-tier input references, **not clean-reference Tier 2 scores**.

## Plan the controlled pilot

```bash
python -m training plan --jobs outputs/training_jobs \
  --config training/configs/4090_pilot.json --output outputs/pilot/plan.json \
  --algorithms sac model_based --seeds 0
```

With 24 prepared jobs this plans 48 runs and starts none. `--warm-models` opts into
transfer within tier/variant/seed; leave it off for the first independent
comparison. Each episode keeps its own actor. Relative paths allow plans to move
with their job/run directory tree.

Explicitly execute a bounded subset when ready:

```bash
python -m training batch --plan outputs/pilot/plan.json \
  --runtime vendor/flash-runtime --start 0 --max-jobs 1 --evaluate
```

Each run uses a separate process to release CUDA/ONNX memory. Batches verify job
hashes and stop on errors; they do not silently retry paid runs or overwrite an
existing run.

## Verification

```bash
python -m unittest discover -s training/tests -v
V2D_PREPARED_JOBS=outputs/training_jobs \
V2D_FLASH_RUNTIME=vendor/flash-runtime \
  python -m unittest discover -s training/tests -v
```

The second command checks each real job with NVIDIA's pinned loader, its 30→50 Hz
timing and the actor/model goal interface. If the submission kit is present at
`vendor/submission-kit/v2d_submission_kit`, the export test uses its actual reader.
Tests cover delayed-control learning, held-out model fit, actor gradients through
a frozen model, SAC-only behavior, checkpoint/tier isolation, replay wraparound,
terminal/reset ordering, evaluation progression, FK, wrench sign/scale, and
quaternion/time export. The synthetic environment is only a test fixture and is
never exported as challenge performance.
