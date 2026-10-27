# Automatic policy training handoff

Run this from a clean clone of `main`. One prepared episode job goes through validation, policy acquisition, precision refinement, full unassisted evaluation, deterministic checkpoint selection and trajectory export without an agent or manual processing decision. `training.scripts.batch` runs the same command over a stable episode roster and splits it across machines using disjoint shard indices.

## Inputs and environment

The input is a portable `v2d_policy_job_v1` directory, containing `job.json`, `motion.parquet`, `reference.npz`, `guidance.npz`, `geometry.npz`, and the object/support assets named and hashed in `job.json`. It describes one rigid object, the native 43-joint G1/Dex3 interface, and a complete episode. Preparing these artifacts from a reconstruction/correction bundle is CPU work:

```bash
python -m training prepare --bundle /data/portable_bundle \
  --urdf /opt/v2d-runtime/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf \
  --output /data/jobs --all --variants corrected
```

The policy command consumes these prepared artifacts. Tier 3 jobs require precomputed motion and an explicit mapping into the released coordinate frame. No clean target is supplied to policy training or checkpoint selection.

Training runs on Linux with one GPU per worker. Clone this repository and obtain the pinned native runtime and its assets:

```bash
git clone https://github.com/Gelrignard/Affordable_V2D_Grounding.git
cd Affordable_V2D_Grounding
git checkout main
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

Run in the native environment, from this repository's root:

```bash
python -m training.scripts.pipeline \
  --job /data/jobs/tier2/episode_000003/corrected \
  --runtime /opt/v2d-runtime \
  --output /runs/tier2/episode_000003 \
  --training-budget-seconds 10800 --total-budget-seconds 11700 \
  --deadline-utc "$DEADLINE_UTC" \
  --dataset-input /data/inputs/tier2 \
  --submission-kit /opt/v2d_submission_kit
```

Set `DEADLINE_UTC` to the actual future ISO8601 UTC deadline for your run before invoking the command. `--prepare-only` runs the same CPU preflight without GPU training. Supply `--parent-state /path/state_N.safetensors` only when reusing a compatible native checkpoint for this exact episode; the command verifies its configuration, trajectory, assets, counters and policy pair. Otherwise acquisition starts from seed 42 with native SONIC and the original assistance curriculum. This acquisition is part of learning the policy; it does not run a separate baseline evaluation.

Omitting both export arguments produces selected policies and evaluation evidence only. Supplying both includes the official-format trajectory export in successful completion.

The fixed processing rule is:

1. Verify committed source, pinned runtime, joint/asset/input hashes, frame count and export inputs; save immutable effective configuration.
2. Train the native acquisition phase to 349,999,104 transitions, under a 9,000-second child cap. A short or failed acquisition stops with an error rather than refining an incomplete parent.
3. Restore the full native learner state, refill fresh replay, and run the same precision recipe for up to 90M more transitions within the remaining episode budget. Save checkpoints every approximately 10M transitions. A clean time-limited refinement can use its valid final checkpoint.
4. Evaluate the final checkpoint and the latest saved checkpoints at or before nominal +60M and +40M, allowing one 2,048-transition collection for native save rounding and never going beyond the final checkpoint. Deduplicate to at most three. All evaluations retain every frame and all 16 worlds, start at exact frame zero, use no reset/global object assistance, and never reset a failed world during an attempt.
5. Select by descending MP-SR, then SP-SR, then ADD-AUC, then ascending MPPE. Ties prefer fewer transitions. This rule uses the input reference only and is saved before evaluation. There is no manual checkpoint choice or parameter sweep.
6. Export fixed world 0 from the selected checkpoint, verify the official trajectory reader and hashes, and mark the episode complete. A successful world is never chosen after observing outcomes.

Defaults retain 2,048 worlds and batch size 2,048, 4M GPU replay, JAX memory fraction 0.60, a 23,000 MiB device-memory guard, and a three-hour training cap. Three bounded evaluations are reserved within the 3.25-hour total cap. Every GPU child enforces both wall time and calendar deadline. A host-level lock and GPU-idle checks prevent a second worker from sharing the device accidentally.

## Batch and teammate shards

Preview the exact roster without GPU work:

```bash
python -m training.scripts.batch \
  --jobs /data/jobs --runtime /opt/v2d-runtime --output /runs/shard_0 \
  --shard-index 0 --shard-count 3 --require-complete \
  --gpu-hours 97.5 --deadline-utc "$DEADLINE_UTC" \
  --dataset-root /data/inputs --submission-kit /opt/v2d_submission_kit \
  --dry-run
```

Remove `--dry-run` to execute. Give each teammate a distinct `--shard-index`, the same `--shard-count`, and the same prepared input roster. The default roster includes all three tiers; `--require-complete` verifies all 30 episode identities per requested tier before any GPU work. Use `--tiers 2 --episodes 3 25` for a subset. `--max-jobs` optionally bounds the selected shard. The example 97.5 hours reserves the full 3.25-hour cap for each of 30 episodes; choose a budget/deadline that matches your actual assigned workload.

Each shard runs serially and accounts for failed work as well as completed work. It stops before an episode if the remaining shard budget cannot cover that episode's remaining allowance. It never rents compute, starts another instance, or submits a leaderboard entry.

## Outputs and restart behavior

Each output directory contains:

- `request.json`, `preparation.json`, stage manifests and configurations: source, input and parent provenance.
- `ledger.json`: all charged GPU wall time, including failed work.
- `bootstrap/` and `refinement/`: policies, compact learner states, metrics and logs.
- `evaluation_N/`: complete rollout arrays, native trajectories, unassisted contract and scores.
- `candidate_plan.json` and `selected_checkpoint.json`: fixed candidates, ranking rule, selected policy/state and SHA256 hashes.
- `exports/tierN/episode_XXXXXX/`: official-format trajectory and export manifest when export arguments are supplied.
- `pipeline_status.json`: `prepared`, `running`, `complete`, or `failed`, including a failing stage and reason.

Repeat the identical command to verify and reuse completed stages. Completed episodes are verified without new GPU use, even after their original deadline. Source, paths, inputs, budgets and deadlines cannot silently change within the same output. The pipeline refuses an active worker, a partial/failed GPU stage or altered evidence; it does not secretly restart training or choose a different method. This is automatic normal processing, with explicit errors for input or infrastructure failures.

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
