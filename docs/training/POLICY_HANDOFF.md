# Automatic policy training handoff

Use one clean committed checkout for all tiers. `training/scripts/run.py` takes an available prepared episode or correction bundle through validation, native policy acquisition, tier-specific refinement, full unassisted evaluation, deterministic checkpoint selection and official trajectory export. Its batch mode uses a stable roster and disjoint shard indices. No agent or manual checkpoint choice is needed.

## Inputs and environment

The input is a portable `v2d_policy_job_v1` directory, containing `job.json`, `motion.parquet`, `reference.npz`, `guidance.npz`, `geometry.npz`, and the object/support assets named and hashed in `job.json`. It describes one rigid object, the native 43-joint G1/Dex3 interface, and a complete episode. Preparing these artifacts from a reconstruction/correction bundle is CPU work:

```bash
python -m training prepare --bundle /data/portable_bundle \
  --urdf /opt/v2d-runtime/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf \
  --output /data/jobs --all --variants auto
```

The policy command consumes these prepared artifacts. Tier 3 jobs require precomputed motion and an explicit mapping into the released coordinate frame. No clean target is supplied to policy training or checkpoint selection.

Training runs on Linux with one GPU per worker. Set `V2D_CODE_COMMIT` to the published commit you intend to run; use that same commit for every tier and teammate. Clone this repository and obtain the pinned native runtime and its assets:

```bash
git clone https://github.com/Gelrignard/Affordable_V2D_Grounding.git
cd Affordable_V2D_Grounding
git checkout "$V2D_CODE_COMMIT"
git rev-parse HEAD
python -m training.scripts.bootstrap --runtime /opt/v2d-runtime --download-assets
```

Git LFS is required for the upstream G1 meshes and SONIC weights. Runtime commit: `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. The native environment setup below supports the pinned CUDA 12 package set or the official frozen CUDA 13 environment on a compatible driver. Native FlashSAC uses JAX, Warp and ONNX Runtime.

Generate and run the installation commands on the destination Linux host:

```bash
python -m training.scripts.setup_runtime --runtime /opt/v2d-runtime \
  --venv /opt/v2d-policy-env --profile cuda12 --driver-major 570 \
  > /tmp/v2d-native-setup.sh
bash -e /tmp/v2d-native-setup.sh
export PATH="/opt/v2d-policy-env/bin:$PATH"
export LD_LIBRARY_PATH="$(python -c "import pathlib,sysconfig; print(':'.join(str(p) for p in sorted((pathlib.Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use the actual driver major reported by `nvidia-smi`. On driver 580 or newer, `--profile cuda13 --driver-major 580` uses the official frozen runtime lock and requires `uv` installed. The setup generator reads the pinned runtime and prints commands; generation itself installs nothing. CUDA 12 pins reproduce the working environment's direct/core package versions; the fresh environment has a CPU import check, while the guarded first episode exercises CUDA/SONIC. All installation happens before episode processing.

Use the organizer's pinned submission kit and the released input tier directory for export. For Tier 1/2, this is the downloaded tier dataset. The examples use `/data/inputs/tier1`, `/data/inputs/tier2` and `/data/inputs/tier3` as aliases for the respective input directories; the batch command expects these names. The official packer's `/data/challenge` argument is a separate root with the organizer's release layout, including `track_2/tier_1_multiview_caption/` or the mirror's `track_2/baseline/` for clean scoring.

For Tier 3, the upstream reconstruction adapter must provide `episode_XXXXXX.npz` with scalar `episode_index`, scalar `coordinate_frame='released_track2_world'`, integer `frame_index=np.arange(T)` and `pose_xyzw` shaped `[T,1,7]`. An `input_frame_manifest.json` binds each NPZ hash, sequence/object identity and the job's source hash, for example:

```json
{"schema":"v2d_track2_input_frames_v1","tier":3,"coordinate_frame":"released_track2_world","files":{"episode_000005.npz":"<NPZ SHA256>"},"episodes":{"5":{"sequence_id":"<job sequence_id>","object_names":["<released object name>"],"source_sha256":"<job source_sha256>"}}}
```

The exporter checks the complete reference transform against `reference.npz`; it never chooses a camera alignment or uses a clean target. The input adapter must establish this mapping upstream; the exporter validates the full trajectory and fails if the mapping is unavailable or inconsistent.

## One episode, one command

Set `DEADLINE_UTC` to the actual future ISO8601 UTC deadline. Run from this
repository in the native environment:

```bash
python training/scripts/run.py \
  --job /data/jobs/tier1/episode_000003 \
  --runtime /opt/v2d-runtime --output /runs/tier1/episode_000003 \
  --deadline-utc "$DEADLINE_UTC" \
  --dataset-input /data/inputs/tier1 --submission-kit /opt/v2d_submission_kit
```

`python -m training run` and `python -m training.scripts.run` are equivalent.
An absolute path to `training/scripts/run.py` also works outside the repository;
relative input/output paths are resolved from your current directory.

`--job` accepts a direct variant directory or an episode directory. Automatic
selection prefers the produced `corrected` job, otherwise uses `original` with
a warning. An explicitly named original remains original. A correction marked
`physical_accepted=false`, a diagnostic candidate, or a reference without contact
labels is usable guidance with its limitations recorded; it is not promoted to
physically accepted ground truth. Corrupt artifacts, inconsistent hashes and
failed correction-bundle stages are errors, not automatic fallback cases.

Add `--prepare-only` for CPU input, native scene and export preflight without
training. Supply `--parent-state /path/state_N.safetensors` only to reuse a
compatible native learner checkpoint for this exact tier and episode. Its input,
physics, optimizer clocks and policy pair are checked. Otherwise native learning
starts from seed 42 with frozen SONIC and the native assistance curriculum.
No separate baseline evaluation is trained or required.

To begin from an existing correction bundle instead of a prepared job:

```bash
python training/scripts/run.py \
  --bundle /data/portable_bundle --tier 2 --episode 3 \
  --urdf /opt/v2d-runtime/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf \
  --runtime /opt/v2d-runtime --output /runs/tier2/episode_000003 \
  --deadline-utc "$DEADLINE_UTC" \
  --dataset-input /data/inputs/tier2 --submission-kit /opt/v2d_submission_kit
```

This performs automatic available-reference selection and CPU job preparation
under `output/prepared_input/`, then runs the same policy pipeline. A preparation
receipt binds the bundle, URDF and resulting job hashes. Repeating the command
verifies and reuses the same preparation; it never overwrites changed or partial
artifacts. This entry starts from prepared motion or a correction bundle. Use the
[correction pipeline](../correction/PIPELINE.md) upstream for raw motion; this
command does not reconstruct video.

## One source revision, tier-specific recipes

`--recipe auto` and `--selection-policy auto` are the entry-point defaults:

| Input | Training lineage | Fixed checkpoint ranking |
| --- | --- | --- |
| Tier 1 | Native acquisition to 349,999,104 transitions, then up to 90M precision refinement | Descending MP-SR, SP-SR, ADD-AUC; ascending MPPE |
| Tier 2/3 | Same native acquisition, 60M precision, then up to 30M supported load transfer | Descending ADD-AUC; ascending MPPE; descending MP-SR, SP-SR |

Tier 1 keeps the existing precision objective and success ranking. Tier 2/3's
final phase uses `supported_load_transfer_v2`: separate partial credit for actual
support load, settling and finger disengagement. The original hand tracking is
retained. The checked input geometry supplies a placement region, not a claim
that imperfect motion or quasistatic correction is correct.

The supported phase currently requires an unscaled object, a compatible native
cylindrical support, and a valid final release window. A typed geometry/window
inapplicability emits a deterministic warning, records its reason, and uses
precision for the final phase. This applies to valid inputs without a supported
cylinder or final release window. A scaled-object contract violation, hash
mismatch, malformed reference or runtime binding error remains a failure.
No support is invented. Acquisition-region
experiments are not enabled by these automatic recipes. Tier 3 uses the same
configuration and input contract but has not been GPU validated.

All tiers use 2,048 worlds, batch size 2,048, 5M GPU replay, the same native
FlashSAC/SONIC action interface and the original optimizer clocks. The default
budget is 10,800 training seconds and 11,700 total GPU seconds per episode,
including evaluation. The native acquisition child has a 9,000-second cap and
must finish before refinement. The Tier 2/3 precision stage must reach its full
60M target before load transfer; a short first stage stops rather than continuing
from a different parent. Tier 1 refinement and the final Tier 2/3 30M stage may
end cleanly at a valid checkpoint under the wall-time cap. The saved status states
what actually ran.

The pipeline fixes candidate checkpoints before evaluation and records their
actual transition counts in `candidate_plan.json`. Native save rounding means a
nominal boundary may select an earlier saved checkpoint; read the recorded
counts rather than assuming an exact 40M/60M/90M comparison. Ties after the four
ranking metrics prefer fewer transitions, then checkpoint hash. A recipe or
ranking override must be explicit and becomes part of the immutable request.
The lower-level `pipeline` and `batch` commands retain their `legacy` defaults
for existing runs; use `run` for automatic tier selection.

Every candidate is evaluated over every frame and all 16 worlds, including
failures, from exact frame zero with no reset/global object assistance and no
within-attempt reset. Selection uses only the supplied tier's input reference.
World 0 is always exported; a successful world is never chosen after outcomes.
Tier 1 export follows the official visible-frame grid while retaining original
frame IDs and timestamps, and the manifest lists every unscored source frame.
Tier 2/3 input visibility cannot redefine a clean evaluation grid. Clean target
scoring is separate from training, checkpoint selection and export.

## Batches and teammate shards

Preview the validated roster without GPU work:

```bash
python training/scripts/run.py \
  --jobs-root /data/jobs --runtime /opt/v2d-runtime --output /runs/shard_0 \
  --shard-index 0 --shard-count 3 --require-complete \
  --gpu-hours 97.5 --deadline-utc "$DEADLINE_UTC" \
  --dataset-input /data/inputs --submission-kit /opt/v2d_submission_kit \
  --dry-run
```

Remove `--dry-run` to execute serially. Here `--dataset-input` is the root
containing `tier1/`, `tier2/`, and `tier3/`; for a single job it is the tier
directory itself. Automatic recipe/ranking selection happens independently for
each job, so one batch may contain all tiers.

Give teammates distinct `--shard-index` values, the same `--shard-count`, the
same source commit and the same input roster. `--require-complete` requires all
30 identities per requested tier before any GPU work. `--tiers 2 --episodes 3 25`
selects a subset; `--max-jobs` caps a shard. The 97.5-hour example reserves 3.25
hours for each of 30 episodes; choose the actual budget and deadline for your
workload. Batch `--dry-run` validates the roster and portable inputs only; use a
single-job `--prepare-only` for native scene/export preflight.

## Outputs, limits and resume

Each episode records:

- `request.json`, preparation and phase configurations: source, inputs, recipe and parent provenance.
- `input_selection.json`: selected reference variant and retained quality warnings.
- `ledger.json`: charged GPU wall time, including failed work.
- Training phase directories: policy/state pairs, actual clocks, metrics and logs.
- `evaluation_N/`: all rollout arrays, unassisted evaluation contract and scores.
- `candidate_plan.json` and `selected_checkpoint.json`: fixed candidates, ranking and selected policy/state hashes.
- `exports/tierN/episode_XXXXXX/`: official trajectory and export manifest.
- `pipeline_status.json`: prepared/running/complete/failed state and any failing stage.

Repeat the identical command to verify and reuse completed work. A completed
run is checked without new GPU work even after its original deadline. Source,
paths, inputs, budgets and deadlines cannot silently change in the same output.
Partial or failed GPU phases do not restart automatically. A host lock and GPU
idle checks prevent overlapping workers; budgets and calendar deadlines apply
to every child. Batch accounting includes failed work and stops before a new
episode when its remaining allowance cannot fit. This command neither rents
compute nor uploads a leaderboard submission.

## Official evaluation packaging

The per-episode export preserves checkpoint and evaluator provenance. Collect finished shards with one CPU command, which verifies manifests and copies only official trajectory Parquets, without choosing checkpoints or worlds:

```bash
python -m training.evaluate.export \
  --collect-runs /runs/shard_0 /runs/shard_1 /runs/shard_2 \
  --kit /opt/v2d_submission_kit --output-dir /results/collected
```

The collection manifest reports missing episodes and provides packing command templates for complete tiers. Use the official packer to check the full roster and recompute clean-reference scores on CPU:

```bash
python /opt/v2d_submission_kit/tools/pack_submission.py \
  --track track_2_tier2 --results /results/collected/tier2 \
  --dataset-root /data/challenge --frame-references /data/inputs/tier2 \
  --code-commit "https://github.com/Gelrignard/Affordable_V2D_Grounding/commit/$(git rev-parse HEAD)" \
  -o /results/track2_tier2.csv
```

Use `track_2_tier1` / `track_2_tier3` and their corresponding input-frame references for the other tiers. Clean-reference scoring happens after policy/checkpoint/world selection is fixed. The CPU packaging command does not retrain, choose checkpoints, change trajectories or upload the CSV.

See [POLICY_METHOD.md](POLICY_METHOD.md) for the objective, curriculum, dynamics and evaluation contracts. The public package contains reusable processing code; run outputs and local investigation logs are not part of it.
