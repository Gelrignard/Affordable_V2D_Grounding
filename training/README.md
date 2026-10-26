# Policy training after trajectory correction

**Preserved CHORD baseline, 26 October 2026:** the current cloud baseline uses
NVIDIA's native JAX FlashCHORD FlashSAC learner, full-body G1/Dex3 simulation,
and frozen SONIC. `experiments/native_chord_diagnostic.py --method native` is
the baseline entry point. It uses our prepared trajectories, a 2M rather than
10M replay, and the pinned native runtime at
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. This is CHORD downstream policy
training, not a new SAC or SuperMimic implementation. The native 350M pair and
subsequent explicitly labeled adaptive/wrench extensions have completed; their
scores and limitations are recorded below. Tier 3 reconstruction is separate.

This version is preserved on `work/tier2-rl` before the user-authorized
DexMachina-style experiments on a separate branch. Historical PyTorch SAC/WM
code remains for reproducibility; its queue is paused and it is not the active
baseline. The next section describes that earlier experimental pipeline.

## Active DexMachina-inspired experiment branch

`work/tier2-dexmachina` preserves the CHORD baseline at commit `fef0ec7` on
`work/tier2-rl`. Its bounded tests are specified in
[`experiments/dex_ppo_tests_20261026.json`](../experiments/dex_ppo_tests_20261026.json).
`experiments/replay_capacity_sweep.py` fills and wraps real FlashSAC buffers,
measures total GPU memory and throughput, and isolates allocator failures.

`experiments/run_dex_ppo_tests.py` waits for the capacity sweep, verifies eight
PPO iterations plus checkpoint evaluation, then runs native CHORD PPO for up
to 20.054M transitions/20 minutes and DexMachina-inspired PPO for up to 60.015M
transitions/45 minutes. Both use corrected Tier 1 episode 5, seed 42, 2048
worlds, 24-step rollouts, native PPO losses and KL-adaptive learning rate,
unchanged full-body/finger SONIC residuals and 50Hz control. Neither uses a WM.
These caps are diagnostic budgets, not promised convergence times.

The inspired variant adapts DexMachina's rolling reward/completion, reward
plateau and recovery ideas to CHORD's seven assistance scales. It starts with
25% exact frame-zero resets, rising to 50% at zero assistance. Reward means
are divided by the native weight and control timestep before gating; completed
episode statistics are weighted by the number of completed episodes. It clears
rolling evidence after each curriculum change. Thresholds, decisions and full
curriculum state are saved. This is an adaptation on the humanoid CHORD
environment, not a reproduction of the floating-hand DexMachina system.

Checkpoint evaluation runs after each learner process exits, avoiding two GPU
allocators competing for VRAM. Every approximately 10M checkpoint plus the
terminal checkpoint receives a zero-assistance, frame-zero, 16-world evaluation
with raw poses, mesh samples and independently cross-checked leaderboard metrics.
The native PPO control retains the native 2000-iteration first assistance
threshold; this short control therefore remains assisted during training. The
20M comparison isolates the proposed curriculum/reset change on PPO, while the
existing FlashSAC 20M/40M/60M checkpoints compare the whole candidate pipeline.
Neither comparison establishes full-budget superiority. AUC-selected checkpoints
are exploratory selections on these evaluation worlds, not held-out scores.

## Historical custom SAC/world-model pipeline

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
  --output outputs/training_jobs_v1 --all --variants original corrected
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

The container recipe follows the pinned runtime's CUDA 13 driver assumption
(host driver >=580). It has not been built locally. The provisioned AutoDL host
instead uses CUDA 12, its existing PyTorch, and Microsoft's CUDA 12 ONNX build;
follow [the measured AutoDL setup](CLOUD_PILOT.md) on that machine. For a separate
dedicated CUDA 12 environment, install the CUDA 12 ONNX build before resolving
the runtime package, then verify on the actual host:

```bash
python -m training.bootstrap --runtime vendor/flash-runtime --download-assets
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps onnxruntime-gpu==1.28.0 \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install vendor/flash-runtime/robotic_grounding/flash_chord
python -m training doctor --runtime vendor/flash-runtime
python -m training.gpu_preflight --runtime vendor/flash-runtime
```

Bootstrap requires Git/Git LFS and fetches G1 meshes and SONIC weights. It refuses
to alter an existing unpinned or dirty runtime. `doctor` reports dependencies,
CUDA and asset hashes; environment construction verifies SONIC hashes and the
clean runtime commit. A 133-byte LFS pointer is not a model. `doctor` is a readiness
report, not a completed simulator smoke test. `gpu_preflight` compiles a CUDA
kernel and runs both SONIC networks, rejecting CPU fallback. The adapter also
recomputes derived robot frame poses from the resampled joints at 50 Hz, so
independent Cartesian interpolation cannot violate the runtime's FK checks.

## Profile, train and evaluate

The measured 4090 config uses 256 worlds and GPU replay. On the provisioned
instance, the episode-16 profile used 4.4 GiB total device memory and reached
about 8,200 transitions/s after warmup, versus 4,900/s with CPU replay. See
[profile scope and results](CLOUD_PILOT.md); this is a throughput measurement,
not a task-success comparison. `replay_device: "cuda"` follows the selected
CUDA device, including when independent jobs use separate GPUs. The generic
`TrainConfig` default remains CPU replay for other machines.

```bash
python -m training profile \
  --job /jobs/tier2/episode_000016/corrected \
  --runtime /opt/v2d-runtime --config training/configs/4090_pilot.json \
  --output /runs/profile --hours 0.1 --transitions 250000

python -m training train \
  --job /jobs/tier2/episode_000016/corrected \
  --runtime /opt/v2d-runtime --config training/configs/4090_pilot.json \
  --output /runs/episode16 --algorithm model_based --hours 2

python -m training evaluate \
  --job /jobs/tier2/episode_000016/corrected \
  --runtime /opt/v2d-runtime --checkpoint /runs/episode16/latest.pt \
  --output /runs/episode16_eval --worlds 1
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

The current pilot is **episode 5 (big black box) and episode 16 (spray bottle)**,
Tier 1/2, original/corrected, seed 0, with the same model-based learner. This is
eight runs. The roster is in `training/configs/rl_pilot_two.json`; it supersedes the
earlier six-episode / two-learner sweep for policy training. The six-episode
correction data remains available. "Original" is our contact-aware retargeting
before correction, not NVIDIA's retargeter. This pilot isolates correction under
one learner; it is not a direct performance comparison against full CHORD.

```bash
python -m training plan --jobs outputs/training_jobs_v1 \
  --config training/configs/4090_pilot.json --output outputs/pilot/plan.json \
  --experiment training/configs/rl_pilot_two.json --seeds 0
```

The planner selects eight of the 24 prepared jobs and requires complete coverage.
By default it uses only the algorithm in the config; an additional algorithm
requires an explicit `--algorithms` override. The 2 h/run and 10M-transition
limits are stop caps, not convergence predictions: at most 16 training GPU-hours,
plus setup/evaluation. Keep score-versus-time and score-versus-transition curves,
and account for correction cost. `--warm-models` opts into
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

Two GPUs can run **independent policies**, one process per GPU. Memory is not
pooled and one policy is not distributed across cards. Launch these in separate
terminals (after profiling; no job starts merely by creating the plan):

```bash
python -m training batch --plan outputs/pilot/plan.json --runtime vendor/flash-runtime \
  --shard-count 2 --shard-index 0 --max-jobs 4 --device cuda:0 --evaluate
python -m training batch --plan outputs/pilot/plan.json --runtime vendor/flash-runtime \
  --shard-count 2 --shard-index 1 --max-jobs 4 --device cuda:1 --evaluate
```

`--start` indexes entries within the selected shard. Shards are disjoint and
cannot be combined with sequential world-model transfer. Evaluation uses the
same GPU as training. On two separate one-GPU instances, use `cuda:0` for both
and preserve the same plan and shard indices.

## Local SONIC wiring and reference viewer

```bash
python -m training.viewer --jobs outputs/training_jobs_v1 \
  --urdf vendor/flash-runtime/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf \
  --port 8766
```

Open http://127.0.0.1:8766. The viewer explains SONIC's encoder, feedback decoder,
residual injection and training loop with links to the pinned code. It includes
an action mixer, the eight-run matrix and budget calculator. The orbitable G1
skeletons are FK previews of real original/corrected references, with object mesh
bounds and source frame IDs. They are not trained SONIC or physical rollouts.
Only the preview is sampled at 10 Hz; training data is unchanged. The server
binds to localhost and serves only generated viewer assets. Ctrl+C stops it.

## Verification

```bash
python -m unittest discover -s training/tests -v
V2D_PREPARED_JOBS=outputs/training_jobs_v1 \
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

## Completed two-episode pilot (2026-10-25)

Eight independent runs used 10,000,128 simulator transitions each, seed 0, the
same 256-world model-based learner, frozen SONIC, and no virtual object support.
The queue took 11,889 seconds (3 h 18 min 9 s); training loops took 22.9–24.7
minutes each. Inputs and code remain preserved in the original cloud `project`
directory. The deployed learner commit was `484cff143df19cec4efdd7e5dd3fc55f232c2bc8`.

Mean object-center position error over a full, unreset evaluation, in metres:

| Tier | Episode | Original | Corrected |
|---|---|---:|---:|
| 1 | 5, big black box | 1.4595 | 0.5495 |
| 1 | 16, spray bottle | 0.5570 | 0.6285 |
| 2 | 5, big black box | 0.6093 | 0.4580 |
| 2 | 16, spray bottle | 0.6325 | 0.6308 |

Original means our retargeting before correction, not NVIDIA's retargeter.
All eight trajectories were finite, but all encountered tracking termination
within 1.72–4.88 seconds; recording continued through the entire episode. A
tracking termination does not itself identify a robot fall. The official
submission-kit SP-SR and MP-SR formulas both yield zero for every run **against
the corresponding tier input reference**. These are pilot diagnostics, not
successful manipulation, statistical significance, or official leaderboard
scores. Correction helps episode 5 here, while episode 16 gives mixed results.

The local artifact `outputs/policy_pilot_two/comparison.json` contains all eight
training times, errors, failure times, and input-reference MPPE values calculated
with the submission-kit math and its first-frame rigid alignment. Evaluation
Parquets and physical traces are beside each run. Full checkpoints/logs remain
on the prepaid cloud instance, which stays powered on.

The challenge scores every tier against the clean object trajectory. Its
[October 23 leaderboard snapshot](https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json)
shows MPPE of 6.72 cm for the leading Tier 1 AUC entry (Chivatam), and 13.70 cm
for the leading Tier 2 AUC entry (Xueyi Wang (FF)). MPPE averages six object-fixed
points at 5 cm radius and includes orientation; it is not center-position error.
Our two-episode, input-reference diagnostics cannot be ranked against those
whole-tier clean-reference scores. Tier 1 clean data is not used to train,
select, or tune Tier 2 policies.

## Longer CHORD-informed training

`configs/4090_full_chord_curriculum.json` and `configs/full_two.json` select four
fresh runs: corrected episodes 5 and 16 in Tiers 1 and 2. Each gets 349,999,104
transitions, matching the pinned G1/SONIC FlashCHORD schedule's budget. This is
our PyTorch SAC/world-model learner with CHORD's assistance curriculum, not a
reproduction of its PPO/FlashSAC optimizer or a direct CHORD comparison.

Virtual object support follows 1, .75, .5, .25, .1, .05, then zero at 60% of the
budget. Exact first-frame starts enter at 82.5% (25%) and 87.5% (50%); remaining
starts sample the reference timeline. Unlike native CHORD's settling resets,
our resets have no freeze period or extra reset-time assistance. Soft corrected
posture targets, live contact/wrench guidance, and frozen SONIC remain in place.
The raw residual bound is 10 and initial normalized action standard deviation
is .05, matching the G1 FlashCHORD action-range/initial-noise recipe. Leg
residuals remain disabled. These changes make this a new training recipe, not a
continuation of the controlled pilot ablation.

Assistance is an explicit policy, critic, and world-model goal input with fixed
[0,1] normalization. Stage changes clear replay, reset the environments, and
require fresh held-out model validation before imagined policy updates resume.
Weights/optimizers continue across stages. Resume uses cumulative transition
counts to recover the correct stage; the curriculum budget remains fixed.

Full, unreset, zero-assistance evaluation runs every 35M transitions and at the
end, in a separate process. Checkpoints are kept every 10M transitions, at stage
boundaries, and for evaluations. A 24-hour cap per invocation preserves a
checkpoint if throughput degrades; reaching that cap is reported as incomplete
and stops the queue after evaluation. At pilot throughput, all four runs are
estimated at roughly 55–60 hours plus evaluation overhead, not a guarantee.
Finishing this transition budget does not establish convergence or task success.

```bash
python -m training plan --jobs outputs/training_jobs_v1 \
  --config training/configs/4090_full_chord_curriculum.json \
  --experiment training/configs/full_two.json --output outputs/policy_full_two/plan.json
python -m training batch --plan outputs/policy_full_two/plan.json \
  --runtime vendor/flash-runtime --max-jobs 4 --evaluate --device cuda:0
```

## Convergence diagnosis, 2026-10-26

The first full run (Tier 1, corrected episode 5) has not demonstrated successful
manipulation. Inspection of saved physical traces changes the interpretation of
the scalar error curve: center error falls from 0.5623 m at 35M transitions to
0.4259 m at 140M, but the robot falls in every evaluated trace. At 140M its root
height falls from 0.792 m to 0.089 m and the box ends at height 0.212 m, versus
0.462 m initially. Contact occurs in only 1.79% of recorded frames. The reference
root height stays between 0.630 and 0.810 m. These are input-reference diagnostics,
not official leaderboard scores. Continued recording after tracking termination
means a finite full-length output is not successful manipulation.

`experiments/diagnose_policy_checkpoint.py` performs bounded inference-only
diagnostics with no optimizer steps. The immutable 160M checkpoint was tested in
four vector worlds: learned residuals, zero residuals, residuals scaled by 0.1,
and finger-only residuals. All four eventually fell. With zero residuals, root
height crossed below 0.35 m at 10.88 seconds; no hand/object contact occurred.
This establishes that learned residuals alone do not explain the failure. It
does not isolate the cause among reference preparation, reset, controller input,
and task interaction. Different worlds can have different actuator delays, so
this is a diagnostic intervention, not a repeated-seed causal benchmark.

On fresh data collected with the checkpoint's 0.1 assistance scale, the WM's
eight-step object-position error was 0.03375 m mean and 0.05181 m at the 90th
percentile. Its weighted normalized state MSE was 0.268, below the readiness
threshold of 1. This gate does not establish contact accuracy, useful action
gradients, or accuracy under zero assistance. Raw compact evidence and the hash
of the executed diagnostic script are in
`experiments/policy_training_diagnosis_20261026.json`; full traces remain at
`/root/autodl-tmp/v2d/diagnosis_20261026` on the cloud instance.

The learner is an experimental SAC/WM hybrid, not a SuperMimic reproduction.
Every collection of eight steps across 256 worlds adds eight SAC updates, four
eight-step WM updates, and one eight-step imagined actor update when ready. It
retains the full simulator budget, so the WM currently adds cost without a
demonstrated wall-time reduction. SuperMimic instead trains its policy through
longer differentiable model rollouts with an imagined-return value learner and
shares a motion-conditioned model/policy across motions. No evidence collected
here establishes a disconnected autograd graph or identifies a single optimizer
bug. Predictive error alone cannot validate the model Jacobian used by the actor.

Several deliberate changes also depart from the native FlashCHORD recipe:
weaker body/palm reference weights, replacement of force closure by a custom
wrench reward that is zero before contact, no settling freeze/reset assistance,
relaxed termination thresholds, and timestamp-resampled reference playback.
These differences need controlled validation; sharing the simulator and SONIC
does not make this a baseline reproduction. The native action bound also permits
large residuals, so that bound alone is not evidence of a coding error.

The inspected [DexMachina source](https://github.com/MandiZhao/dexmachina/tree/adae5bf620c57723d185b2757ee3ce9656927c20)
offers concrete mechanisms to borrow: `envs/curriculum.py` reduces assistance
after reward/episode-length criteria and can dial it back after regression;
`envs/rewards.py` supplies geometric approach guidance before contact forces
appear; `envs/robot.py` uses bounded wrist residuals with joint-limited finger
targets; `rl/train_rl_games.py` uses the existing vectorized PPO learner. Its
floating-hand task and throughput are not direct G1/4090 performance promises.

Recommended order before another long run: diagnose the zero-residual
reference/reset/controller interaction; establish a faithful, memory-scaled
FlashSAC downstream baseline on one episode; retain dense pre-contact guidance
and assess assistance stages using unassisted behavior; only then test whether
the WM reduces time to the same physical task outcome. Use early fall/contact/
object-lift checks in addition to trajectory error, and compare learned versus
zero residuals before expanding episode count or parallelism. The diagnosis
does not isolate or reject the trajectory corrector's contribution. No training
settings or queue contents were changed by these diagnostic runs.

## Checkpointed pause and executed diagnostics, 2026-10-26

The user subsequently authorized a pause and active diagnosis. The trainer
handled SIGINT and saved `latest.pt` at **172,564,480 transitions**, including
actor, critics, WM, optimizers and counters. The scheduler was stopped before
interrupting its child, so none of the three pending runs began. The checkpoint
SHA-256 is `c1177bee7d9ed7d9755861755c7f9575f18dd2e3aeced8746c3bba8811588776`;
its contents were verified unchanged after diagnosis. Resume restores those
states with fresh replay. The prepaid instance remains powered on. The status
endpoint now distinguishes this verified intentional pause from failure.

`experiments/policy_pause_diagnosis_20261026.json` preserves 18 configurations
of four-world zero-residual physical tests, five component/throughput profiles,
the pause record, and executed-source hashes. Full traces and disposable
profiling outputs remain under `/root/autodl-tmp/v2d/diagnosis_20261026`.
Diagnostics used separate output directories and never updated the paused
checkpoint. Profiling performed optimizer steps only in disposable copies.

Measured throughput on the **same RTX 4090**, with FP32 and equal SAC training
examples per collected transition:

| Worlds / SAC batch | SAC plus WM transitions/s | SAC only transitions/s |
|---|---:|---:|
| 256 / 256 | 7,392 | 9,878 |
| 1,024 / 1,024 | 24,272 | 32,622 |
| 2,048 / 2,048 | 40,591 | 51,762 |

Each timing window used 32 collections after warmup. At 256 worlds the
simulator/adapter consumed about 47% of measured wall time, SAC updates 23%,
and WM fitting plus imagined actor updates 25%. The profiler observed many
small kernel launches. Skipping already-fitted normalizers was immaterial;
TF32 did not improve these timings. The 2048-world profile reserved 3.90 GiB
in PyTorch; this excludes Warp/ONNX allocations and is not total GPU memory.
These are throughput measurements, not matched convergence experiments.
The WM batch also scales with the SAC batch; imagined-actor batch remains 128.
The SAC-only measurement follows other disposable updates, so its behavior
distribution is not a fresh, independently trained policy. Extrapolating
350M transitions at 51,762/s gives 1.88 h, excluding setup/evaluation and any
throughput changes during learning; it is not a time-to-success promise.

Physical diagnosis isolated a reference-consumption issue. Episode 5's original
and corrected inputs have identical root/leg motion. Sending the corrected
whole-body motion to SONIC caused all four worlds to fall around 10.6–11.2 s.
Original motion delayed the fall to 21.6–21.8 s. Native environment construction,
settling resets and source-frame playback did not remove the corrected-input
failure. Holding the reference clock at frame zero stayed upright for the
16-second check, but this repeats the frame-zero future encoder window; it is
not a fully static-reference or manipulation success test.

The new explicit `sonic_reference_mode: original_body` keeps the corrected
task/finger references and policy guidance while taking only SONIC's body-motion
tokenizer input from the same episode's original reference. Timeline, joint
names, tier, episode and source identity are checked, and the carrier hash is
included in checkpoint compatibility. The default remains `task`, preserving
old checkpoint semantics. On the real adapter this separation delayed episode
5's fall to 21.7–22.0 s. Episode 16 improved less consistently: one of four
worlds remained above the 0.35 m root threshold for its 15.87 s duration, versus
zero of four previously. These small diagnostic tests do not establish task
success, an official score, or a general improvement across episodes.

Freezing the original leg reference or root orientation separately did not
remove the later fall. Thus a constant leg/orientation shortcut is not supported
by these tests. Our full-run configuration masks all leg residuals, whereas
native G1 FlashCHORD exposes them; restoring limited whole-body corrective
authority is a sensible next controlled experiment. Restoring useful approach
rewards and performance-gated assistance should be assessed with the same
physical checks before another long queue. The added carrier option is a
partial reference-interface fix, not a convergence fix by itself.

Future evaluations now report minimum root height, first time below a labeled
0.35 m diagnostic threshold, first tracking failure, and measured hand/object
contact fraction alongside trajectory error. The threshold is not an official
fall/success definition. These fields make a fallen robot visible without
requiring manual trace inspection. The original full queue remains paused;
no replacement long training run was launched during diagnosis.

## Native CHORD baseline and 4090 memory tests, October 26

The follow-up tests use the pinned upstream `scripts/train_flash_sac.py` entry
point and its native JAX FlashSAC learner. They retain G1 ReconBody's reward,
termination, settling reset, observation, exploration, n-step replay, mixed
precision, four updates per collection, and full 29-body-joint residual action
space. This is a downstream CHORD baseline on our prepared motions. It is not
our PyTorch SAC with the world model switched off, nor a reproduction of
NVIDIA's upstream retargeting. The original full SAC/WM queue remains paused.

`experiments/native_chord_diagnostic.py` attaches a local JSONL metric sink to
the upstream entry point, checks replay allocation before training, polls total
GPU memory with `nvidia-smi`, and bounds wall time. It never edits the pinned
runtime. `experiments/run_native_chord_pair.py` runs original and corrected
inputs with identical settings, then evaluates each in a separate process to
release training allocations. A native checkpoint/evaluator replaces our `.pt`
checkpoint format for these tests. Prepared support assets and the verified
LF/CRLF variants of the same G1 URDF are resolved explicitly.

The reference is fixed during training. SONIC's encoder reads future body-joint
positions/velocities and root orientations; its decoder reacts to the current
robot state. The learned body residual is added after that decoder, before
joint-specific scaling. Fingers use reference joint targets plus their own
learned residuals. Neither the reference nor SONIC's weights are progressively
rewritten by RL. The paused corrected run supplied the precomputed corrected
body reference. Holding the clock at frame zero was a diagnostic intervention,
not its training configuration. Native CHORD permits leg residuals; our paused
run masked them. The `original_body` carrier option is a separate experimental
deviation and is not enabled in this native baseline.

The native environment is isolated at `/root/autodl-tmp/v2d/native_chord_env`.
It reuses existing physics dependencies/assets while adding pinned JAX 0.10.2,
Flax 0.12.7, Optax 0.2.8, and native evaluation dependencies. CUDA 12 wheels,
cuDNN 9.10.2.21 and NVCC 12.8.93 support the installed driver 570; the published
CUDA 13 environment requires a newer driver. No driver or original training
environment was changed. Most downloads used the Tsinghua mirror. The small
TFP nightly evaluation dependency unavailable there was fetched from the
upstream lockfile URL and its SHA-256 verified.

Measured native runtime probes used Tier 1 episode 5's original reference:

| Worlds | SAC batch | Replay capacity | Peak total GPU memory | Outcome |
|---|---|---|---|---|
| 1,024 | 2,048 | 1M | 8,459 MiB | Completed 2,000,896 transitions |
| 2,048 | 2,048 | 4M | 23,887 MiB | Memory guard interrupted; no CUDA OOM reported |
| 2,048 | 2,048 | 2M | 15,691 MiB | Completed 2,000,896 transitions |

The successful probes' median post-warmup throughput was approximately 25,475
and 42,970 transitions/s, respectively. The latter left about 8.7 GiB of device
memory free at the sampled peak. This includes JAX, Warp/physics and ONNX Runtime,
unlike the earlier PyTorch-only counter. A one-second sample is not a guarantee
against every allocation spike; the tighter subsequent preflight also budgets
2.5 times logical replay bytes plus 4 GiB for workspaces. Do not run two of the
2M-replay native learners concurrently on this GPU. The native full 10M replay
recipe remains unsuitable without further memory work.

For the bounded original/corrected comparison, each policy gets 10,000,384
transitions, 2,048 worlds, batch 2,048 and replay 2M. The native 350M learning-rate
horizon and fixed assistance thresholds remain unchanged; all of this short
budget therefore uses assistance scale 1.0. Native source-frame playback is also
retained. These settings differ from the earlier timestamp-resampled adapter,
so compare original versus corrected within this pair, not its absolute errors
against the old pilot as if all evaluation conditions were identical.

Evaluation uses the native evaluator, sixteen worlds, explicit frame-zero
initialization, no object assistance and no within-episode resets. Its object
errors and CHORD reference-tracking success are measured against the tier's
input reference, not official clean-ground-truth leaderboard scores. Finite
output and reaching the final recorded frame do not imply successful control.
Cloud source, checkpoints, logs and evaluations are retained under
`/root/autodl-tmp/v2d/diagnosis_20261026/native_pair_t1_ep5`. The status endpoint
reports this diagnostic queue separately from the intentionally paused full
queue.

The pair completed in **769.2 seconds (12.8 minutes)** including initialization,
compilation, checkpointing and both evaluations. Results are preserved in
`experiments/native_chord_diagnosis_20261026.json`:

| Tier 1 episode 5 input | Steady training transitions/s | Peak total GPU memory | Unassisted object-center error | Native reference success |
|---|---|---|---|---|
| Original | 42,894 | 15,693 MiB | 0.6242 m | 0/16 |
| Corrected | 43,983 | 15,701 MiB | 0.5549 m | 0/16 |

Both variants have exactly identical object positions and orientations. Each
uses one training seed and sixteen evaluation worlds, not sixteen independent
trained policies. Both cohorts violate pelvis, palm and object reference
criteria. The corrected mean object error is lower by 0.0693 m in this short
test, but neither policy achieves successful manipulation. Occasional mixed
precision update skips were small (maximum logging-window fractions: 1% actor,
0.5% critic), and the last windows had none; they do not establish an optimizer
failure. Training's force-closure contribution is nonzero, so its live contact
reward path is active.

At this measured sustained throughput, 350M simulator transitions project to
about **2.2–2.3 hours** before initialization and evaluations. This is a runtime
projection, not a convergence prediction. The native recipe does not remove
assistance until about 210M transitions and introduces deployment-exact starts
only at about 289M. Consequently these 10M probes establish runtime feasibility
and an early matched comparison, not that the native recipe has converged or
failed to converge at its intended budget. The fixed-reference SONIC design is
also used by CHORD and is not by itself evidence of an implementation error.

Before changing the curriculum, run the native learner for its full intended
budget. The 10M tests do not establish failure at 350M. Earlier deployment starts,
performance-gated assistance and a conservative body reference remain possible
later ablations, not fixes established by these short tests. Reintroduce a WM
only when it reduces time to the same physical outcome. The paused checkpoint
SHA-256 remains
`c1177bee7d9ed7d9755861755c7f9575f18dd2e3aeced8746c3bba8811588776`.

### Authorized full-budget pair, October 26

The user authorized fresh Tier 1 episode-5 original and corrected policies,
sequentially, with 349,999,104 training transitions **each** and a rolling replay
capacity of 2M. Replay capacity is a learning-parameter departure from the native
10M recipe; it is not the total training budget. Both use the native seed 42,
2,048 worlds, batch 2,048, optimizer, rewards, resets, full-body action space,
reference playback and unchanged assistance curriculum. There is no WM.

The queue runs from `/root/autodl-tmp/v2d/native_full_20261026`, with results in
`pair_t1_ep5`. The runner saves native actor and learner-state checkpoints every
10,000,384 transitions and at the final step. It allows four hours of wall time
per policy, retains the existing total-device-memory guard, and stops the queue
if training or evaluation fails. The first policy's finished training allocations
are released before native unassisted evaluation over sixteen worlds. Each
policy also gets a bounded 1,100-frame native-viewer MP4; rendering errors are
recorded without discarding a policy or cancelling the second training run.
The viewer can restart failed episodes; quantitative evaluation does not reset
within an attempt. Final errors remain against tier input references.

The short-run throughput projects to about 2.2–2.3 training hours per policy;
allow 5–6 hours for the pair including setup and evaluation. This is not a
convergence guarantee. The old custom queue and all earlier artifacts remain
preserved. The prepaid instance stays powered on. Scheduled summaries occur
only at 00:00, 06:00, 12:00 and 18:00 America/New_York.

The pair completed at 08:02 New York time on October 26, taking 16,806.6 seconds
(4 h 40 min) including evaluations and the initial unsuccessful video attempts.
Both policies reached 349,999,104 transitions and the final zero-assistance
curriculum stage. Full results are in
`experiments/native_chord_full_results_20261026.json`.

| Tier 1 episode 5 input | AUC ↑ | SP-SR ↑ | MP-SR ↑ | MPPE (cm) ↓ | Training time |
|---|---|---|---|---|---|
| Original | 0.3184 | 0.0000 | 0.0000 | 23.3374 | 2 h 20 min |
| Corrected | 0.2546 | 0.0000 | 0.0000 | 28.5571 | 2 h 18 min |

The separate native reference-tracking success diagnostic was 0/16 for both.
Object-center errors were 0.2324 m and 0.2846 m; those are not MPPE.

These are unassisted evaluations against the tier input reference, with one
training seed per variant. The corrected input has higher final object error
and lower ADD-AUC in this test; neither policy satisfies the native success criterion.
This single episode does not establish a general effect of correction, and the
reduced replay remains a departure from the published recipe. Peak total GPU
memory was 15,689 and 15,693 MiB, respectively; training completed without a
memory-guard stop. No additional training has been started.

Initial MP4 export failed because pyglet tried to connect to an X11 display on
the headless host. Render subprocesses now select `PYGLET_HEADLESS=true` and
`PYOPENGL_PLATFORM=egl` before importing the renderer. Video recovery uses only
the terminal policies; it does not retrain or change the saved evaluations.

Both recovery exports completed and were downloaded with evaluation reports as
one checksum-verified 2.3 MB bundle. The clips are 22 seconds at 50 FPS and
1280 × 720, available under
`outputs/cloud/native_full_t1_ep5/{original,corrected}/policy_headless.mp4`.
The viewer may restart failed episodes; use the separate no-reset evaluation
reports for quantitative results. Video hashes and render times are recorded in
the result JSON. That completed pair's monitor was paused after reporting. The
subsequent candidate experiments and their monitoring are described below; the
instance remains powered on.

## Leaderboard-style reporting

Future native pairs and custom evaluations automatically write `leaderboard.json`,
`leaderboard.csv` and `leaderboard.md`. The cloud status endpoint exposes these
values separately from assistance rewards and native reference-tracking success.
Reports use the challenge's column order, four decimal places and 0–1 scales for
AUC/SP-SR/MP-SR; MPPE is in centimetres. Missing or invalid scores are unavailable,
not zero. Coverage, evaluation worlds, training time, transitions and reference
identity remain visible. No rank or full-tier score is inferred from one episode.

The four definitions follow the released Track 2 scoring code:

- AUC integrates ADD accuracy at thresholds 1–9 cm. Local mesh samples are used;
  the host's hidden sample set is not distributed.
- SP-SR applies the 10 cm / 0.5 rad thresholds to each world's time-averaged
  object pose error, without centering or initial alignment.
- MP-SR requires each object's mean pose error below 3 cm and 30 degrees.
- MPPE averages errors of six object-frame points at ±5 cm along the axes,
  over points, bodies, worlds and frames. Native `mean_per_point_error_cm` is
  used instead of the legacy multi-body maximum in `mppe_cm`.

Numerical tests compare our CPU formulas with the released Kaggle numerical
functions. The completed native pair's SP-SR, MP-SR and MPPE were also recomputed
from its saved Parquets without a GPU; MPPE agrees to better than 1e-4 cm. Its
already saved native AUC is retained. Raw evaluations and checkpoints are unchanged.

Backfill a paired result record without simulation or retraining:

```bash
python -m training.leaderboard \
  --results experiments/native_chord_full_results_20261026.json \
  --output outputs/cloud/native_full_t1_ep5
```

Custom evaluation also preserves `auc_vertices.npz`, enabling later CPU rescoring
with exactly its local sample set. A nonfinite world invalidates the local cohort
score rather than being silently discarded. Tier input references remain distinct
from the clean references used by official scoring.

## Native CHORD extension tests (26 October)

The structured design and controlled test sequence are in
`experiments/chord_dynamics_training_proposal_20261026.json`. Its first controlled
implementation is `experiments/chord_candidates_v1_20261026.json` and
`experiments/run_chord_candidates.py`. The user authorized implementation and
bounded comparisons. The old custom SAC/WM queue remains paused.

Two fresh seed-42 policies use corrected Tier 1 episode 5: adaptive curriculum
alone and the same curriculum plus a measured object-wrench reward. Native
FlashSAC, frozen SONIC, all residuals, optimizer horizon, and native rewards are
preserved. Each uses 2048 worlds, batch 2048, a 2M replay, at most 60,002,304
transitions, and a 30,001,152-transition stall gate. The adaptive condition also
introduces prescribed frame-zero starts earlier; it is a curriculum package,
not an isolated assistance-only ablation. Existing native checkpoints supply
the baseline curve without new baseline training.

Every 10,000,384 transitions, save policy and optimizer state, then run a full
16-world unassisted evaluation and a separate next-lower-assistance probe.
Only the physical probe gate advances assistance; assisted reward is never the
gate. All four metrics are independently recomputed from the saved poses and
local mesh samples before publication. Positive-assistance probes never write
a leaderboard table. Missing/invalid scores remain unavailable. Checkpoints
do not include the replay buffer, so they are not bit-exact resume snapshots.

The wrench term uses smoothed object-COM dynamics and actual simulator inertia,
subtracts measured non-hand support, and measures the remaining hand wrench.
Virtual object forces are excluded. Contact friction and actuator limits come
from the simulator. Grasp contact presence sums individual force magnitudes,
so opposing finger forces do not cancel out the engagement indicator. This
version scores realized wrench; it does not solve a new contact-force QP or
certify the corrected trajectory. The native reward terms stay unchanged.

An integration run verified training, saving, unassisted/probe evaluation and
independent scoring within 16.2 GiB total GPU memory. CPU tests exercise force
reaction signs, torque about the COM, gravity supported by the floor, quaternion
sign invariance and the curriculum gate. Learning results are reported
separately from these implementation checks. PPO, a phase-state bank, joint
prior reweighting and the optional WM are not enabled in this initial ablation.

The bounded pair was launched from commit `8b6151a` in
`/root/autodl-tmp/v2d/native_candidates_20261026_v1`. Its immutable source snapshot
is under `candidate_pair/executed_source`; logs and outcomes remain alongside it.
The read-only endpoint exposes `candidate_training` separately from the paused
custom queue and completed native baseline. Summaries ran at 00:00, 06:00,
12:00 and 18:00 New York time without intervening checks. The monitor is now
paused after both terminal candidate outcomes were reported. The prepaid
instance remains powered on. A planned stall-gate stop is recorded as a
scientific outcome rather than a queue crash.

Keep native CHORD, frozen SONIC and all body/finger residuals. First isolate a
DexMachina-inspired competence-based assistance curriculum, then add a robust
required-wrench target derived from object dynamics and support forces. Corrected
joint trajectories remain soft priors. Check native PPO against FlashSAC only
with a shared environment/reward/curriculum contract. An optional shared compact
WM must pass contact-transition, gradient and total-cost tests before affecting
the policy. It does not justify resuming the old per-episode full-state WM queue.

## Completed candidates and input/configuration audit (26 October)

Both candidates stopped at the planned 30,001,152-transition limit without a
curriculum advance. Training assistance remained 1.0; the separate 0.75-assistance
probes failed the gate, including pelvis-reference tracking in all 16 worlds.
That tracking failure is not itself a measurement of falling. The numbers below
are the separate full, zero-assistance evaluations:

| Method at 30M transitions | AUC ↑ | SP-SR ↑ | MP-SR ↑ | MPPE (cm) ↓ | Training (min) | Evaluation (min) |
|---|---|---|---|---|---|---|
| Adaptive curriculum | 0.1106 | 0.0000 | 0.0000 | 66.0019 | 13.17 | 4.35 |
| Adaptive plus wrench | 0.1157 | 0.0000 | 0.0000 | 47.0797 | 13.12 | 4.34 |

The queue, including baseline-checkpoint scoring and setup, took 36.64 minutes.
At matched 20M transitions, the native checkpoint had AUC 0.1169 and MPPE
68.9902 cm; adaptive had 0.1645 / 42.0925 cm and adaptive plus wrench had
0.1514 / 40.6224 cm. Both candidates regressed at 30M, and all success rates
remained zero. This establishes neither successful control nor a convergence
speedup. Full rows, timings and terminal diagnostics are preserved in
`experiments/chord_candidates_v1_final_results_20261026.json`.

The audit of all four cloud-saved configurations (the earlier native 350M pair
and these two candidates) confirms identical optimizer, network, action and
simulation settings. The configured learning-rate range is 3e-4 to 1.5e-4,
with the unchanged native 683,592-update horizon, four updates per collection
and an actor update every second optimizer attempt. Actor and critic use
separate clocks: the saved 350M checkpoint has 682,772 optimizer attempts and
341,386 actor attempts, so its next critic rate is about 1.50e-4 and its actor
rate about 2.25e-4. The 30M candidate rates remain about 2.974e-4 and 2.993e-4.
Control is 50 Hz with four physics substeps. Replay capacity
is 2M rather than the native recipe's 10M. The candidates additionally change
the curriculum, early frame-zero starts and stopping rule; only the wrench
candidate changes the objective. No world model or PPO was used.

Two input-contract findings limit the comparison with correction replay:

- The hash-verified cloud input is Tier 1 episode 5, box rotation, from the
  automatic October 24 bundle. Its manifest labels it
  `diagnostic_pre_spline_candidate`, with `physical_accepted: false`. The word
  "corrected" in the result tables denotes that variant; it must not imply a
  physically accepted optimized demonstration. The specific replay the user
  viewed has not been identified from the current message.
- Native G1 `source_frame_playback: true` preserves all 1003 source frames but
  consumes one per 50 Hz control step. The source timestamps span 33.40 seconds
  at 30 Hz, while the training reference spans 20.04 seconds. This native
  setting therefore changes physical timing relative to the correction replay;
  its contribution to learning failure has not been measured. SONIC reference
  velocities also use the 50 Hz playback rate.

The correction replay additionally uses a supported pelvis and observed-force
feedback; importing position references does not transfer that controller.
The user clarified that the trajectory remains useful for learning and its
strict upstream acceptance flag is not a training eligibility gate. The
upstream SOMA/G1 exporter also defaults to 30 fps without resampling; our
50 Hz source-frame playback matches the released G1 trainer. This timing
difference from local correction replay is therefore not an established
explanation of the score gap to NVIDIA.

The next proposed test keeps that input and native timing, and uses native
CHORD PPO with its KL-adaptive learning rate and a DexMachina-style
reward/completion assistance curriculum with recovery. The v1 pelvis-reference
probe gate is not carried over. The bounded proposal, which has not been
launched, is `experiments/next_no_wm_training_proposal_20261026.json`.
This audit did not launch further training or alter any executed experiment.
Evidence and configuration hashes are in
`experiments/chord_configuration_input_audit_20261026.json`.

## Dex-inspired PPO continuation (26 October)

The short Dex-inspired native PPO candidate completed 60,014,592 transitions in
1005.56 training seconds, versus 1494.41 seconds for the saved approximately 60M
FlashSAC baseline. Its terminal local AUC was 0.1546 and MPPE 47.59 cm, versus
0.1452 and 61.93 cm. Both had zero SP-SR and MP-SR. This supports a longer test,
not a claim of successful manipulation. The assistance gate had not advanced.

The user authorized continuation to 350,011,392 total transitions, matching the
previous full FlashSAC comparison budget. `experiments/dex_ppo_full_20261026.json`
records the plan. Native PPO's default 20,000 iterations with 4096 worlds and
24 rollout steps instead total 1.966B transitions. The continuation uses the
tested 2048-world configuration and restores the complete native learner plus
curriculum stage and evidence history. Native resume resets simulator state.
No optimizer, reward, gate or controller setting changes. All approximately 10M
checkpoints remain; roughly 100/150/200/250/300/350M receive separate unassisted
evaluation with independently verified leaderboard-style metrics.

CHORD includes both PPO and FlashSAC. PPO uses a fresh 49,152-transition rollout
batch here, followed by five epochs and four minibatches; it has no rolling 5M
replay buffer. FlashSAC capacity tests passed 5M through 7M with about 23.3 GiB
total GPU usage, while 8M encountered CUDA OOM. This establishes observed fit,
not a learning improvement or enough room for overlapping evaluation.

## Reward audit and native FlashSAC 5M comparison

The 350M PPO continuation improved assisted step reward from 0.0756 to 0.1322
(first/last 40 iterations), but kept assistance at 1.0. Final unassisted AUC
fell to 0.1158 from 0.1989 at 300M. Recomputing all six checkpoint evaluations
from raw poses reproduced all four metrics. No PPO updates were skipped and
the weighted reward components sum to the total within 2.5e-8. The late episode
endings were dominated by palm-tracking failure, keeping the completion and
progress gates closed. This implicates the assistance handoff in this adaptation;
it does not isolate PPO or the trajectory correction as the cause.

The adaptation uses per-step reward statistics rather than DexMachina's
episode-return/max-length statistics. Its thresholds, discrete gain levels,
reset mixture and whole-body controller also differ. The native object reward
tracks position directly, while AUC depends on both position and rotation.
These findings and their limits are in
`experiments/ppo_reward_method_audit_20261026.json`.

The user requested a fresh native FlashSAC run with 5M replay, batch 2048,
2048 worlds, 349999104 transitions, the same corrected Tier1 episode5/seed42,
and unchanged native optimizer/reward/reset/curriculum. The launcher rejects
learner configuration differences from the saved 2M baseline other than replay
capacity and checks the motion hash. It retains native 10M checkpoint intervals
and evaluates seven checkpoints after training releases GPU memory. Specification:
`experiments/flash_sac_5m_20261026.json`. This run uses neither the Dex-inspired
curriculum nor a world model.
