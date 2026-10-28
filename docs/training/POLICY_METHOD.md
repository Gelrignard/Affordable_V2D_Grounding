# CHORD and the fixed precision refinement

This implementation uses the pinned NVIDIA G1+Dex3 ReconBody FlashSAC runtime
at `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`, frozen SONIC, and the existing
43-dimensional body/finger residual action interface. Each episode has its own
policy. It is not one policy trained to solve every episode.

## Original training and the changes

| Mechanism | Pinned native CHORD recipe | Fixed refinement |
|---|---|---|
| Learner | Native FlashSAC, native actor/critic, optimizer, observation/action interface | Preserved |
| Controller and physics | Frozen SONIC, G1/Dex3, Newton/Warp contact dynamics | Preserved |
| Parallelism | 2,048 worlds, batch 2,048 | Same 2,048/2,048 |
| Replay | 10M GPU transitions | 5M GPU transitions, native storage/sampling/update semantics |
| Initial learning | Native 349,999,104-transition assistance curriculum | Same initial learning; no separate baseline run in production |
| Object position reward | Broad exponential with 0.20 m scale, weight 1 | Broad/fine mixture, final weight 3 |
| Object orientation | Used in termination; no explicit orientation tracking reward term | Adds orientation reward, final weight 2, 0.6 rad scale |
| Hands | World-space reference palm position/orientation tracking | During reference contact, track the hand pose relative to the live object; otherwise retain world-space guidance |
| Contact reward | Live force-closure coverage gated by reference contact labels, weight 5 | Same contact signal, final weight 1 |
| Tracking termination | Immediate palm/object threshold crossing | Brief recovery window plus gross-failure limits; pelvis checks preserved |
| Evaluation | Separate rollout/evaluation path | Exact frame 0, zero global/reset assistance, no settling or within-attempt reset, training tracking termination disabled |

The current branch reduces replay capacity from 10M to 5M,
disables JAX preallocation, uses memory fraction 0.60, and monitors total device
memory against 23,000 MiB. Batch size and world count remain 2,048, matching this
pinned native recipe. The replay capacity reduction preserves the native batch size and world count.
Actual memory use, throughput and convergence depend on the episode and GPU;
the pipeline records these values and enforces its configured bounds.
Earlier precision and acquisition trials used 4M replay. A new 5M run therefore
changes replay capacity as well as any explicitly enabled objective variant.

## Why the hand change matters

Suppose the object is a few centimeters away from its reference position while
the robot still holds it correctly. A strict world-space hand target can pull
the hand toward the reference location and away from the live object. During a
reference-contact phase, the new objective instead compares

\[
R_o^T(p_h-t_o)\quad\text{with}\quad
\hat R_o^T(\hat p_h-\hat t_o).
\]

The left side is the actual palm in the actual object's frame; the right side
is the reference palm in the reference object's frame. Contact labels are
smoothed over 10 frames to blend between world and object-relative guidance.
A 3 cm position tolerance and softer palm orientation reduce conflict with noisy
retargeted hands. Labels remain a reference prior; they do not prove a real grasp.

## Why the precision reward matters

With object position error e in meters, the refinement uses

\[
r_p=\tfrac23\exp[-(e/0.20)^2]
    +\tfrac13\exp[-(e/0.05)^2].
\]

The broad component still guides recovery from a large error. The fine component
distinguishes centimeter-scale differences that the original broad reward barely
separates. Over the first 10M refinement transitions, object-position weight
ramps 1→3, object-orientation weight 0→2, and force-closure weight 5→1. Body-joint
tracking remains weight 5. The new raw orientation term is included in native
SAC replay reward relabeling, so it changes learning rather than only log scores.

Refinement restores actor, critic, target networks, optimizer state, reward
normalizer and schedule clocks, but starts with fresh replay filled by that
restored policy. Ordinary rollout virtual object control is zero. The fixed
50/50 immediate-frame-zero/recovery-start mixture is retained; recovery starts
may retain the native brief stabilizer. A short palm/object error can be recovered
from before termination, while gross divergence and native pelvis checks remain.
No such assistance or training termination is allowed to shorten evaluation.

## What dynamics contributes

Actions pass through SONIC and the robot's actuators into contact simulation.
Mass, inertia, gravity, friction, contact forces, body balance and collisions
determine whether an action moves, holds, slips or drops the object. SAC learns
from these executed transitions. The native force-closure term encourages useful
contact geometry/capability; the refined tracking objective rewards the resulting
object trajectory. This is physics-based policy learning, not kinematic replay.

The precision recipe does not add an explicit residual between required
object wrench and feasible hand/support forces. It does not solve a friction-cone
force projection or jointly optimize a trajectory and its forces. These are separate possible dynamics objectives, rather than claims made by this
implementation. A successful simulator rollout does not certify a noisy reference
or an upstream correction as dynamically feasible.

## Optional Tier 2/3 acquisition regions

`--acquisition-guidance grasp-region` changes palm-position guidance during a
short approach window before a reference contact onset. It constructs an
object-local palm offset from nearby contact-labelled frames whose fingertips
are close to the object's surface and whose palm has exterior clearance. A
bounded region around that offset moves with the simulated object, so the hand
can approach the object where it actually is without chasing every corrupted
world-space palm sample. The approach region blends back into the existing
contact objective at the onset. Orientation, contact reward, held-object
guidance, release, object rewards and SAC remain unchanged.

The input motion and object mesh provide candidate grasp geometry. Observed
frame masks, seed consistency and mesh clearance reject unsuitable windows;
rejected windows retain the original guidance. These checks concern palm and
fingertip geometry, not whole-hand collision freedom or force feasibility. Real
grasp stability must still emerge from simulated contact dynamics and the
resulting SAC transitions. No clean target or successful rollout is used to
construct the regions. The configuration, source reference and executed region
table are hashed alongside the checkpoint.

This is an experimental, opt-in Tier 2/3 variant inspired by separating grasp
acquisition from demonstration tracking. It is not a DexMachina reproduction
and does not change to PPO. The default precision profile, including Tier 1,
retains the previous objective and execution path.

## Optional supported placement and release

`--placement-guidance supported-release` keeps the precision approach, grasp and
transport objective. It detects a final reference contact ending near a nearly
stationary, geometrically supported object target. CPU preflight checks the input
mesh, inertial center of mass and the native support transform. It retains both
object and hand targets and rejects incompatible windows. The current runtime
adapter requires one unscaled object and one axis-aligned cylindrical support.

Within the accepted window, a bounded bonus rewards actual object-support load,
low object velocity and a pose near the placement target. The bonus also favors
removing hand load after support is established. While the object lacks stable
support, only the withdrawing hand's tracking cost is softened, to no less than
20% of its original value. That relief ends with the window; time continues and
full hand tracking returns. This does not add virtual support forces, freeze
the motion clock or modify the physics, controller or policy interface.

The changes remain in the native 13 raw reward terms before n-step aggregation,
so replay contains the executed shaping. Evaluation records support force and
individual hand-link contact slots; its actor does not use the reward. The
input reference/support sidecars and their hashes must accompany the checkpoint
when this diagnostic objective is constructed. The variant is opt-in for Tier
2/3 and cannot be combined with `grasp-region`. Geometry checks are not a
dynamic-feasibility certificate; full unassisted rollouts determine its outcome.

## Method sources and scope

CHORD supplies the reference-tracking formulation and contact-capability ideas;
its whole-body path explicitly uses a reduced force-closure objective for noisy
reconstructions. FlashSAC supplies the native off-policy learner. Object-relative
hand shaping, the fine position mixture and the recovery thresholds here were
chosen from rollout diagnoses and bounded experiments, not copied as a complete
recipe from one paper.

- [CHORD, sections 3.1–3.2](https://arxiv.org/html/2607.00033v2#S3)
- [FlashSAC](https://arxiv.org/abs/2604.04539)
- [Native implementation](https://github.com/nvidia-isaac/video_to_data/tree/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/robotic_grounding/flash_chord)


The combination of object-relative hand shaping, precision reward and recovery
thresholds is an empirical modification of the native method. It is not an exact
reproduction of a separate paper. Per-episode evaluations diagnose the resulting
policy; they do not establish generalization across the full challenge.

The implementation is in `training/core/refinement.py`; configuration assembly
and checkpoint-name compatibility are in `training/core/configuration.py`.
`training/evaluate/policy.py` records uninterrupted simulation trajectories,
`training/evaluate/metrics.py` computes local input-reference diagnostics, and
`training/evaluate/score_clean_reference.py` applies the pinned official CPU
metrics against clean references after input-only coordinate conversion.
