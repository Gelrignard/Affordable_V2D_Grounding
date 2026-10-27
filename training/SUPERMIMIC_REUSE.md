# SuperMimic reproduction and checkpoint audit

Inspected 25 October 2026, while the eight-run V2D pilot was training. These are
independent implementations of [SuperMimic](https://supermimic.github.io/), not
an assertion that the original authors have released these implementations.
Reported benchmark results have not been independently reproduced here.

| Source | Useful material | Acquired revision |
| --- | --- | --- |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | MuJoCo Warp/mjlab learner, differentiable kinematics and rewards, imagined actor-critic updates | `d9be385b3eeb13a638fb33fb200d3bfd9be9839f` |
| [Baojch/SuperMimic_Dyn](https://github.com/Baojch/SuperMimic_Dyn) | Adaptation of CuboiLeo; public full checkpoint and documented training/evaluation provenance | `28b15acb68ab3871f2ae1479964d53457ac066c6` |
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | Isaac Lab implementation with a learned dynamics model and motion-tracking learner | `a1508d42c9d1699a52e8233ab16d08c488441ab9` |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | UniLab implementation, environment/learner tensor contract, full-clip evaluation | `faf7818e8abac3ec6abb8f00e6e904d3577f800b` |
| [lzyang2000/morphmimic](https://github.com/lzyang2000/morphmimic) | Multi-robot dynamics/policy sharing and morphology conditioning | README reviewed; sparse source checkout timed out, not recorded as acquired |

The first four source snapshots are under `/root/autodl-tmp/v2d/research/`,
alongside `repositories.json`. They were downloaded without installing their
training environments or large motion libraries into the running V2D environment.

## Verified public checkpoint

The [SuperMimic_Dyn model card](https://huggingface.co/unpackableorange/SuperMimic_Dyn)
publishes `ckpt_0040000.pt`, including the world model, policy, critic,
optimizers, configuration and counters. The downloaded artifact is pinned to
revision `669542a05cbef8debd6d008348286db253e3ffe2`:

- Size: 221,362,447 bytes.
- SHA-256: `182b7a5d05b6c3e4364d6223d1cf52ba18e6573563e961f6e019d6d445340c3c`.
- Download: China mirror, 69.8 seconds, while our GPU training continued.
- Inspection: CPU `torch.load(weights_only=True)` with only `pathlib.PosixPath`
  additionally permitted for configuration metadata. No repository training
  code was executed, and no weights were injected into the active V2D runs.
- Evidence: `research/weights/{manifest.json,audit.json,MODEL_CARD.md}` on the
  server; a small local copy is under `outputs/cloud/research_evidence/`.

The tensor shapes and source agree: its dynamics network takes 133 inputs
(71 body features, 29 direct actions, 33 domain parameters), uses three
2048-wide hidden layers, and outputs 70 corrections. Its state contains the
floating base and 29 joints. It integrates velocities and orientation with
explicit time stepping and quaternion operations; rewards/observations use
analytic kinematics. Its robot configuration removes plastic hand geometry and
does not model our finger joints or manipulated object. The model card labels
weight licensing `other`; the source's MIT license is not asserted for weights.

## Relation to the running learner

Our current learner fits a separate model for every pilot entry from executed
simulator transitions. It combines SAC updates with eight-step differentiable
imagined actor updates after a held-out dynamics validation gate. SONIC is frozen.
This is a SuperMimic-inspired learner, not a reproduction of the paper's full
method or its large shared motion library.

The V2D state has 1,996 scalars, including body/finger/object physics, contacts,
SONIC history and delayed actions. The action is a 43-dimensional residual
interface with leg residuals masked out. The public checkpoint's 29 direct
joint commands and 71-dimensional state have different semantics; reshaping
its weights would not make it predict this closed-loop system.

The promising reuse path is a **body dynamics prior plus a learned manipulation
residual**. Convert actual post-SONIC body targets into the public model's joint
order and PD-command units, validate predictions against our simulator, and fit
the missing finger/object/contact effects on our collected transitions. PD gains,
inertias, frame conventions and actuator delays must be included in this mapping.
The prior would be retained only if it reduces held-out prediction error or
sample requirements; this has not yet been measured.

Several ideas transfer without depending on those weights: predict compact
physical state, integrate rotations explicitly, advance known history buffers
deterministically, and calculate kinematics/reward terms analytically wherever
available. This can reduce what the dynamics network must learn. Sharing such
a model across episodes is a further experiment; it must not mix original and
corrected pilot arms or tier data invisibly.

These are concrete follow-on candidates. The current eight entries keep the
same learner, budget and independent initialization throughout, so their
original/corrected comparison remains interpretable. No claim of compute or
task-performance improvement from the downloaded checkpoint has been established.
