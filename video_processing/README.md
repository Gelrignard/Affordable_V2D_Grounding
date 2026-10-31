# Video processing and reviewed tracking repair

This package reconstructs human/object motion from an RGB interaction video,
human/object masks and an object mesh. It runs pinned CARI4D stages 01–06
(MoGe 2, SAM 3D Body, depth alignment, FoundationPose and CoCoNet), then optionally
converts native MHR output into the repository's 77-joint SOMA NPZ format.
It stops before native contact refinement and policy training.

`processing/` and `correcting/` each contain a `run.py` entry point and
`input/`, `core/`, `output/` **code packages**. Keep videos, weights, caches,
review images and generated trajectories under an ignored `outputs/` directory
or external storage, not inside those code packages.

The included correction is one reproducible method: select an already plausible
reconstructed object pose by reviewing an intermediate RGB frame and its overlay;
refine forward and backward from that anchor using the current RGB and estimated
depth; select actual image/depth-supported candidates over the requested interval.
No clean 3D trajectory, Tier 1 pose, future reference motion, interpolation or
synthetic held-pose bridge is an input to this repair.

One reviewed frame can be enough for a short, visible interval with useful masks
and depth. It is not a guarantee: occlusion, symmetry, incorrect monocular depth,
bad masks or an incorrect anchor can require more anchors or a new reconstruction.
This implementation deliberately rejects poor depth support. It does not include
the historical RGB-only wrong-depth trials, manual directed landmarks or carrying-hand
inference used in particular archived episodes. Archived cases retain their own
provenance; this entry point does not claim to regenerate every historical variant.

## Runtime

GPU inference needs Linux, NVIDIA CUDA, the external model assets and the native
CARI4D dependencies. CPU `--help` is model-free; contract checks require NumPy and
SciPy. The code does not download datasets, credentials or checkpoints implicitly.

Use a clean checkout at exactly:

```bash
git clone https://github.com/nvidia-isaac/video_to_data.git /path/to/video_to_data
git -C /path/to/video_to_data checkout 193382a184308f6800331f3db829f76dc4294cfe
```

Prepare its CUDA runtime and weights using the pinned
[CARI4D instructions](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_cari4d/README.md)
and [Docker dependency recipe](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_cari4d/docker/Dockerfile).
The repository source itself is external. Install its reconstruction modules from
that checkout into the GPU interpreter; in particular, editable installs point to
`reconstruction/modules/v2d_cari4d/lib`, `v2d_sam3d_body/lib`, `v2d_sam2/lib` and
their upstream dependencies. Build the FoundationPose native extensions in that
checkout according to the upstream recipe. A host wrapper installation alone is
not a GPU runtime. The native runner verifies that its imported CARI4D module
comes from `--upstream-root`; mounting an unrelated image's module is rejected.

The audited external runtime used Python 3.11, Torch 2.5.1/cu124,
NumPy 1.26.3, nvdiffrast 0.3.3 (the real OpenGL rasterizer),
Kaolin 0.18.0 and `py-soma-x==0.2.1`. The upstream recipe contains additional
compiled dependencies and model license/access requirements. See
[`runtime_sources.json`](runtime_sources.json) for source pins; it is not a full
transitive package lock or a claim that every future dependency resolution works.

Expected weight layout is the upstream combined weights root, containing
`cari4d/`, `sam3d_body/`, `foundationpose/nvlabs_pytorch/`, `hf_home/`, and (for
prompt propagation) `sam2/`. SOMA conversion additionally needs the SOMA assets at
`soma/466879a83d57eabf3d875ded2d869f2075f90348/`. Checkpoint access and downloads
are explicit setup steps performed with the applicable model licenses. No token
is accepted in a task JSON or embedded in these source files.

Set these shell variables to your local paths and run from this repository root:

```bash
UPSTREAM=/path/to/video_to_data
WEIGHTS=/path/to/weights
VIDEO=/path/to/clip.0.color.mp4
MESH=/path/to/object.glb
RUN=outputs/video_processing/run1
```

Use the original supplied GLB at its established metric scale. An uncalibrated
single-image mesh needs separate scale estimation; this pipeline does not infer
or silently fit object scale. Native `.pkl`/`.pth` inputs use Python serialization;
load only files from your trusted local run. The compact output NPZ is pickle-free.

## Masks and reconstruction

If correct packed masks already exist, pass them directly to `reconstruct`.
Otherwise provide SAM2 points or boxes in original RGB pixel coordinates.
Human `object_id` is `0`; manipulated object `object_id` is `1`. Prompt frame
indices are zero-based. Additional prompts may be placed at later frames.

Example `prompts.json` (replace coordinates after inspecting your own video):

```json
{"prompts": [
  {"object_id": 0, "frame_index": 0,
   "points": [{"x": 230, "y": 280}], "point_labels": [1]},
  {"object_id": 1, "frame_index": 0,
   "box": {"x0": 380, "y0": 300, "x1": 430, "y1": 360}}
]}
```

```bash
python -m video_processing.processing.run masks \
  --upstream-root "$UPSTREAM" --weights-root "$WEIGHTS" \
  --video "$VIDEO" --prompts /path/to/prompts.json --frames 900 \
  --output "$RUN/masks"

python -m video_processing.processing.run reconstruct \
  --upstream-root "$UPSTREAM" --weights-root "$WEIGHTS" \
  --video "$VIDEO" --masks "$RUN/masks/masks.h5" --mesh "$MESH" \
  --frames 900 --fps 30 --output "$RUN/native" --stop-after tracking
```

Replace `900` and `30` with the exact decoded frame count and constant video rate.
No retiming is performed. Inspect the human and object masks before expensive
reconstruction; generation does not claim visual acceptance. The file name must
be `SEQUENCE.0.color.mp4` for the pinned upstream interface. The native directory
above is `$RUN/native/clip/` because the sequence is `clip`.

Omit `--stop-after tracking` to run through CoCoNet directly. The default result
is `native/clip/native_coconet.npz`, its provenance JSON and retained native files.
Rerunning unchanged inputs reuses upstream stage markers; changed inputs require
a fresh run directory. The adapter applies a narrow missing-centroid diagnostic
serialization fix in memory. It never edits the pinned checkout or allocates
stage 07 contact refinement. Run `--dry-run` first to validate the local input and
source contracts without loading GPU models.

## Select an anchor and make a candidate

Render likely anchor frames from your own reconstruction:

```bash
python -m video_processing.processing.run inspect \
  --upstream-root "$UPSTREAM" --weights-root "$WEIGHTS" \
  --native-root "$RUN/native/clip" --sequence clip --frames 420,450 \
  --output "$RUN/anchor_inspection"
```

View the RGB overlays yourself. The overlay shows silhouette and object axes;
check translation, extent and orientation, including symmetry ambiguities.
Choose a frame where the existing reconstruction is plausible. `inspect` writes
image hashes and the baseline SHA256 in its `report.json`, but does not mark any
image as reviewed. No 3D pose is entered manually.

Create `repair_plan.json`, replacing hashes, paths and observations with your
actual evidence. `review_image` is relative to the plan file, or an absolute path.
Intervals have **inclusive** endpoints. This example tracks from frame 450 in
both directions; additional anchors can be added to the same interval.

```json
{
  "schema": "v2d.reviewed_anchor_repair.v1",
  "baseline_sha256": "REPLACE_WITH_BASELINE_SHA256",
  "anchors": [{
    "id": "middle", "frame": 450,
    "review_image": "anchor_inspection/review/frame_000450.jpg",
    "review_image_sha256": "REPLACE_WITH_IMAGE_SHA256",
    "reviewer": "operator",
    "note": "Replace with specific RGB observations establishing a usable reconstructed pose."
  }],
  "intervals": [{"start": 400, "end": 500, "anchor_ids": ["middle"]}]
}
```

```bash
python -m video_processing.correcting.run \
  --upstream-root "$UPSTREAM" --weights-root "$WEIGHTS" \
  --native-root "$RUN/native/clip" --sequence clip \
  --plan "$RUN/repair_plan.json" --output "$RUN/repair1" --dry-run

# Execute the same command without --dry-run after validating the plan.
```

The adapter measures baseline and tracked candidates against visible masks and
actual estimated depth. Bounded temporal costs rank **discrete** candidates;
there is no interpolation, averaging or automatic smoothing. A branch stops
after unsupported refinements. Unresolved frames retain the baseline pose with
explicit uncertainty. All native arrays outside the repair intervals are
preserved. Input video, masks, depth, mesh, calibration and review evidence are
hash-bound and checked again after execution.

Outputs are `candidate.pkl`, `plan.json`, candidate measurements in `report.json`,
and baseline/candidate overlays under `review/`. Exit code `2` means the candidate
was saved but failed numeric screening. A successful exit is not visual acceptance
or proof of 3D accuracy. Persistent failures require better input evidence or more
reviewed anchors; do not approve a failed attempt by changing a sidecar.

## Review, run CoCoNet and convert

Actually view all generated review frames (listed in `required_review_frames`):
every 30th target frame, interval endpoints, anchors, and first/middle/last of
each unresolved run. The complete unresolved list stays in the numeric report;
this is sampled visual coverage, not review of every frame. For a usable candidate, create a separate
`repair_review.json` with these fields and your actual observations:

```json
{
  "schema": "v2d.repair_visual_review.v1",
  "candidate_sha256": "REPLACE_WITH_CANDIDATE_SHA256",
  "decision": "accept",
  "reviewer": "operator",
  "viewed_frames": [400, 430, 450, 460, 490, 500],
  "notes": "Replace with actual observations and remaining uncertainty."
}
```

The example frame list is illustrative: use the exact report's required frames,
and only state that you viewed frames you actually inspected.

```bash
python -m video_processing.processing.run reconstruct \
  --upstream-root "$UPSTREAM" --weights-root "$WEIGHTS" \
  --video "$VIDEO" --masks "$RUN/masks/masks.h5" --mesh "$MESH" \
  --frames 900 --fps 30 --output "$RUN/native" \
  --repair-candidate "$RUN/repair1/candidate.pkl" \
  --repair-review "$RUN/repair_review.json"
```

Stages 01–05 reuse the original run. CoCoNet reads the reviewed candidate and
writes a separate branch under `native/clip/repaired/<candidate-hash-prefix>/`.
The original FoundationPose output and earlier CoCoNet results remain intact.
CoCoNet does not automatically understand repair reliability sidecars; inspect
its final human/object reconstruction before treating it as useful downstream
input. The new compact file is that branch's `native_coconet.npz`.

SOMA conversion requires an explicit **rigid** `world_from_camera` matrix mapping
OpenCV camera coordinates in metres to your chosen Z-up world. Supply a separately
measured/manual calibration; no scale fitting or default identity is applied.
The JSON schema is:

```json
{
  "schema": "v2d.camera_world_calibration.v1",
  "world_from_camera": [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],
  "source": "Replace with your camera/world calibration method and measurements",
  "notes": "Illustrative axes only; replace this matrix with the measured transform.",
  "benchmark_frame_indices_used": []
}
```

The example matrix is not a measured camera calibration. For a benchmark-specific
frame-zero alignment, externally compute the rigid transform using only the
permitted first-frame reference, label its source and set
`benchmark_frame_indices_used` to `[0]`. `processing/output/geometry.py` includes
the audited no-scale Kabsch utility on 15 body landmarks. This optional
coordinate alignment is separate from tracking correction. Neither command
fetches Tier 1 data; correcting has no alignment-reference or clean-pose input.

```bash
python -m video_processing.processing.run convert \
  --upstream-root "$UPSTREAM" --weights-root "$WEIGHTS" \
  --native /path/to/chosen/native_coconet.npz --mesh "$MESH" \
  --calibration /path/to/calibration.json --output "$RUN/soma"
```

`soma/human_motion.npz` contains 77 joint positions/orientations, complete
timestamps, original-mesh object poses, `hand_contact_logits`, a relative mesh
path and provenance. Both human and object receive the same world transform.
`camera_soma_motion.npz` retains the decoded camera-space result and fitting
residuals. The final NPZ is consumed by this repository's
`v2d_retarget.dataprocessing.readers.load_npz` and the existing retargeting entry
point's `--input` option. It is marked `training_ready: false`; format validity
does not certify physical contact or policy performance.

## Checks and provenance

```bash
python -m video_processing.check_contracts
python video_processing/processing/run.py --help
python video_processing/correcting/run.py --help
```

The CPU checks cover the stage boundary, both propagation directions, unsupported
branch stopping, no interpolation, depth rejection, preserved frames, schemas,
calibration and a synthetic SOMA-output roundtrip through the real repository
reader. The GPU decoder is mocked in that format test. This new packaged entry
point has not been GPU-rerun; the adapted numerical/native API code comes from
the separately audited runs. See [`PROVENANCE.md`](PROVENANCE.md) for exact scope
and attribution. Native model code/weights remain external and retain their
respective licenses.
