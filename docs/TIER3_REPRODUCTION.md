# Tier 3 reproduction and provenance

This release supports the same robotic-grounding implementation for all three
Track 2 tiers. Tier 3 additionally uses reconstructed human/object motion as its
input. Each tier is packed separately; use one final release commit URL
consistently within each submission. Evaluation and checkpoint hashes remain
the actual hashes of the artifacts used.

## Video input preparation

See [video_processing](../video_processing/README.md) for the runnable commands
and input schemas. The package separates two operations:

- `processing/` runs the pinned CARI4D vision stages through CoCoNet and converts
  their output to the human/object motion interface consumed by retargeting.
- `correcting/` provides one reproducible repair method using reviewed image or
  reconstructed-pose anchors and bidirectional object tracking. This is video
  reconstruction repair, distinct from the robot dynamics correction in
  `correction/`.

Repair input must come from the video, masks, depth, supplied mesh, or the
episode's own reconstructed poses. It does not consume clean Tier 1 trajectories
or clean intermediate-frame poses. A clear intermediate frame can supply an
anchor; heavily occluded or ambiguous intervals may require additional anchors.
An anchor is evidence for initialization, not a promise that every inferred
frame is accurate.

The public repair entry point documents one common method. Historical recovered
inputs also include specialized interventions, such as inferred initialization
during occlusion. Publishing the common entry point does not relabel those
historical inputs or claim that one annotation reproduces every prior repair.
For policy reproduction from an archived input, retain its original quality
labels, annotation records, source hashes, and preferred-variant manifest.

## Coordinate alignment is separate from repair

The conversion interface accepts an explicit calibrated rigid transform and
its provenance. A manual calibration must establish a metric camera-to-world
transform in the desired frame; an image-space box alone is insufficient.

For the challenge campaign, a single rigid transform was estimated from 15
reconstructed first-frame human landmarks and the corresponding Tier 1 frame-zero
human landmarks. The latter were decoded using that frame's human pose, shape,
scale and bone-length parameters. No scale was fitted. The same transform was
applied to the reconstructed human and object sequence. The supplied support
plane established the scene convention.

The benchmark loader downloaded the Tier 1 Parquet bytes to extract the first
row. Later clean motion samples were not used to build or repair the Tier 3
training references. The official FAQ permits first-frame Tier 1 alignment:
https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf

## Reproduce grounding from saved inputs

Preserve the recovered `human_motion.npz`, original mesh, conversion report,
calibration, and source/review manifests. Use the existing retargeting, dynamics
correction, portable-bundle export, and training preparation commands described
in [the correction guide](correction/PIPELINE.md) and
[the policy handoff](training/POLICY_HANDOFF.md).

Training uses the prepared Tier 3 job and its reconstructed reference. Clean
future trajectories are reserved for official scoring after policy selection;
they must not enter training, reference repair, or selection. Submitted object
poses come from the learned policy's simulated rollout.

## Source identity and recovery

The release builds on `040f84f471fa58b7ec9ba8fee95cb446a5d3db7e`, which adds explicit
native checkpoint recovery and handling of non-finite diagnostic telemetry.
The earlier campaign ran `927f24e86b108306ef818e2c1a5ae119ba2ba7e6`, using native
runtime `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. Existing outputs retain those
execution records. A common final release URL identifies the reproducible
implementation; it does not rewrite the source identity inside historical
checkpoints or receipts.

Recovery uses a separate output directory, binds the original failed attempt
and checkpoint/state pair, and retains its canonical charged-time ledger.
Optimizer and curriculum counters are restored; replay storage and simulator
state start fresh. Record the parent checkpoint and recovery lineage rather than
describing it as an uninterrupted run. Completed episodes are reused without
retraining. Use explicit budget and deadline flags when running a bounded
campaign, because the new command-line defaults make these limits optional.

Raw videos, model weights, reconstructed arrays, checkpoints, credentials, and
machine-specific orchestration belong outside Git. Keep verified archives and
the environment/model/input manifests alongside the reproduction records.
