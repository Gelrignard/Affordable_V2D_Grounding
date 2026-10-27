# G1 / Dex3 contact-aware retargeting

This module generates kinematic robot references from human/object motion. It
includes the pinned NVIDIA SOMA-to-G1 body-only objectives as a baseline and a
contact-aware variant with moving fingers, object-relative contact anchors,
hand collision probes, and velocity bounds between saved configurations.

## Run

From the workspace root, using the installed CPU environment:

```text
python retargeting/scripts/run.py --export-parquet
python retargeting/scripts/visualize_viser.py
```

The viewer is at http://127.0.0.1:8081. Select `baseline` or `contact_aware`,
scrub frames, and use **Focus on hands**. Cyan is the human, orange the targets,
magenta the robot fingertips, and green inferred object contact anchors. Red
lines show target-to-FK residuals. Object playback is prescribed, not simulated.

For a faster development comparison:

```powershell
python retargeting/scripts/run.py --stride 4 --output retargeting/outputs/stride4
python retargeting/scripts/visualize_viser.py --results retargeting/outputs/stride4
```

`--stride` retains actual timestamps. `--fps 30` is an explicit timing override
for the example only. The example declares 200 FPS; its original capture rate
has not been independently verified. Viewer playback initially uses 30 FPS for
inspection and does not change the output timestamps.

Use `--method baseline` or `--method contact_aware` for one solver. `--start`,
`--stop` (exclusive) and `--max-iterations` support experiments. Use different
output directories for experiments; matching filenames are replaced on rerun.

## Module boundaries

```text
scripts/                       CLI and visualization entry points
src/v2d_retarget/dataprocessing/ HumanMotion, source adapters and coordinates
src/v2d_retarget/core/           targets, contact inference, IK, diagnostics
src/v2d_retarget/output/         JSON/NPZ and NVIDIA motion_v1 export
configs/                        explicit contact/solver parameters
outputs/                        ignored generated artifacts
```

The workspace uses an ignored `upstream/` Git checkout. Follow the
[optional dependency guide](../docs/NVIDIA_COMPATIBILITY.md) to create it at the
tested revision and apply the compatibility patch. The CPU setup uses Conda Pinocchio/Pink rather than
the upstream Linux-oriented pip installation.

## Inputs

Default: the pinned NVIDIA snack-box parquet, verified against its Git LFS
SHA-256 before cached pickle deserialization. Its 77 SOMA joints are restored
to their recorded world translation and converted to a common Z-up frame.

`--input path/to/human_motion.npz` accepts a pickle-free canonical input:

| Field | Shape / convention |
|---|---|
| `joint_names` | J strings, SOMA joint semantics |
| `positions` | T × J × 3, metres in a common world |
| `rotations_wxyz` | T × J × 4, world orientations, unit quaternions |
| `timestamps` | T seconds, strictly increasing |
| `object_positions`, `object_wxyz` | T × 3, T × 4, same world |
| `object_mesh` | Mesh path; relative paths resolve against the NPZ |
| `coordinate_frame`, `units` | `world_z_up`, `metres_seconds` |
| `sequence_id`, `metadata_json` | String ID and JSON provenance |

Every run writes `human_motion.npz` illustrating this contract. Invalid/missing
observations are rejected. The current solver requires one rigid object and
the named SOMA body/hand joints. Arbitrary challenge LeRobot ingestion,
articulated-object contacts and reconstruction denoising remain separate adapter
work; they are not silently approximated by this solver.

## What changes relative to NVIDIA's SOMA path

NVIDIA's `soma_to_g1.py` maps human Hand joints to wrist-yaw link origins and
does not optimize fingertip targets. After IK, it creates binary proximity labels
using object mesh vertices and palm/human fingertip points. Those labels do not
feed back into its IK. NVIDIA also supplies a separate MANO-to-Dex3 retargeter;
it is not connected to that SOMA whole-body path.

Our body targets retain NVIDIA's ankle/knee/pelvis/torso/wrist mapping. Foot
targets remain useful references for motion tracking. Execution and balance
belong to downstream SONIC + residual-policy simulation; no walking policy is
run or trained here.

The hand stage:

1. Adds six fingertip frames using fixed offsets from NVIDIA's standalone
   Dex3 URDFs. These are distinct from distal-link origins.
2. Extracts human thumb/index/middle tips as soft free-space targets.
3. Finds closest mesh-triangle points for potential contacts. A conservative
   vertex-distance bound rejects distant points cheaply. Enter/exit hysteresis
   uses elapsed seconds, object-relative speed and maximum anchor drift.
4. Latches likely contacts in object coordinates so targets follow the recorded
   object without sliding across its surface. Release/regrasp can relatch;
   blending avoids instant target jumps.
5. Increases contact fingertip costs and jointly solves fingers, arms and body.
   Wrist tracking remains a soft prior; morphology differences can increase
   wrist error as the robot changes its grasp.
6. Penalizes sphere-probe penetration into the object's convex hull. This is
   conservative for open/concave reconstructed objects and is not complete
   mesh or whole-body collision checking.
7. Bounds every internal IK iterate relative to the previous OUTPUT frame,
   using the actual timestamp interval and 90% of URDF joint velocity limits.
   Internal solver iterations cannot accumulate into an over-speed saved step.

The contact representation is inspired by object-centric interaction and
temporal persistence in [HumanEgo](https://github.com/TX-Leo/HumanEgo). This is
an independent implementation: it imports no HumanEgo code, does not modify the
object trajectory, and does not treat inferred contacts as measured forces.

## Outputs and evaluation

- `baseline.npz`, `contact_aware.npz`: q, targets, inferred contacts, human/object
  references, final FK diagnostics and iteration/stop-reason metadata.
- Matching `.json`: root pose, named joint angles, timestamps, object reference,
  contact anchors and metrics, with units and quaternion ordering.
- `*_metrics.json`, `comparison.json`: post-correction body error, contact-anchor
  error, exact contact mesh distance, object-relative slip, hull-probe
  penetration and frame-to-frame joint speeds.
- `motion_v1/sequence_id=.../robot_name=g1/data.parquet`: optional NVIDIA export
  for the cached example. Palm/link/site poses and finger angles are recomputed
  from final q. Both real distal-link origins and virtual tips are named so
  downstream code does not confuse them. Detailed contact hypotheses remain
  in JSON/NPZ; no force/contact manifold is fabricated.

Independent re-evaluation:

```powershell
python retargeting/scripts/evaluate.py retargeting/outputs/snack_box/contact_aware.npz
```

The Parquet is a reference for the supplied `SonicG1-ReconBody-v0` training/eval
stack; retain access to the referenced NVIDIA assets. Canonical inputs currently
export JSON/NPZ because training export also needs scene metadata. Schema/FK
verification here is not an Isaac Lab rollout test.

Inspect `comparison.json` before claiming improvement. Better contact geometry
on one demonstration does not establish better challenge policy success. Check
wrist error, remaining penetration, contact slip, joint limits and timing. The
known knee/pelvis mismatch from human/G1 proportions remains in the body targets.
Floating-base speed, acceleration, torque, friction, support forces and dynamic
balance are not constrained by this kinematic solver.

## Measured example result

On all 800 frames of the pinned snack-box example, at its declared 200 FPS,
using the default configuration and identical inferred contact episodes:

| Metric | NVIDIA body baseline | Contact-aware |
|---|---:|---:|
| Mean active fingertip-to-anchor error | 83.4 mm | 14.0 mm |
| Mean active fingertip-to-mesh distance | 35.8 mm | 4.2 mm |
| Mean object-relative contact slip | 0.606 m/s | 0.070 m/s |
| Saved transitions exceeding joint speed limits | 6 | 0 |
| Maximum hand-probe hull penetration | 91.0 mm | 21.2 mm |

This trades wrist accuracy for hand contact: mean left/right wrist target
errors increase from 2.3/2.2 mm to 19.5/18.1 mm. At least one hand probe still
penetrates the hull by more than 5 mm in 247 frames. The unconstrained floating
base reaches 7.10 m/s at the declared timing. These are kinematic measurements
on one example, not evidence of stable walking or successful physical grasps.
The full report is `outputs/snack_box/comparison.json`; the saved-pose/FK and
Parquet round-trip checks are in `outputs/snack_box/verification.json`.

References: [Track 2](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/),
[NVIDIA V2D](https://github.com/nvidia-isaac/video_to_data),
[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter),
[GMR](https://github.com/YanjieZe/GMR).
