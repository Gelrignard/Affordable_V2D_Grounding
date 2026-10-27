# Trajectory correction

This package refines a retargeted G1/Dex3 manipulation reference through contact
geometry, force allocation, and local robot trajectory optimization. It exports
the source and corrected references with named joints, coordinate transforms,
contact guidance, and separate planning/physical status.

Read the [pipeline description](../docs/correction/PIPELINE.md) for contact
grounding and dynamics assumptions, the [phase interface](../docs/correction/INTEGRATION.md)
for explicit physics inputs, and the [artifact contract](../docs/correction/RL_PILOT_ARTIFACT_FORMAT.md)
for downstream policy preparation.

## Full episode

Dataset decoding and retargeting are separate stages. Install their environments
using [SOMA setup](../docs/SOMA_SETUP.md) and [CPU setup](../docs/SETUP.md).
Full-episode correction additionally needs
`setup/correction-simulation-requirements.txt` and the G1/Pinocchio runtime.

The following commands illustrate one explicit Tier 2 episode. Run decoding
with the SOMA environment and retargeting/correction with the robot environment:

```text
python -m data_tools.fetch_sample --tier 2 --episode 3
python -m data_tools.decode_tier1 --tier 2 --episode 3 --all-frames --data correction/data/tier2 --output outputs/episodes/tier2/episode_000003/human_motion.npz
python retargeting/scripts/run.py --input outputs/episodes/tier2/episode_000003/human_motion.npz --method contact_aware --output outputs/episodes/tier2/episode_000003/retargeted
python -m correction.scripts.correct_rl_pilot_episode --input outputs/episodes/tier2/episode_000003 --output outputs/episodes/tier2/episode_000003/correction
```

The correction output directory must be new. The command keeps the full source
timeline and labels rejected stages and finite fallbacks. Its supported-body
replay and assumed object properties are diagnostics; downstream whole-body
policy training uses the separate pinned runtime.

To export a set of completed episodes, place `selection.json` in the input root
with the intended roster, for example `{"tiers": [2], "episodes": [3]}`. Each
`tierN/episode_XXXXXX` directory contains the decoded `human_motion.npz`,
`retargeted/contact_aware.npz`, and `correction/` outputs above. Existing
preparation reports and batch audits are preserved when supplied.

```text
python -m correction.scripts.export_rl_pilot --root outputs/episodes --output outputs/references
python -m correction.scripts.validate_rl_pilot outputs/references
```

The `rl_pilot` command names and `v2d_rl_pilot_v1` schema remain for compatibility.
`original_reference.npz` means this project's contact-aware retargeted reference.
The object trajectory remains the same-tier observed target. An exported
candidate does not become a physically validated demonstration: read
`planning_accepted`, `physical_accepted`, and `reference_kind` in its manifest.

## Explicit contact phase

The smaller phase solver accepts a geometric reference and an explicit JSON
configuration describing contact patches, frame conventions, mass, inertia,
friction, and force limits. See the [phase interface](../docs/correction/INTEGRATION.md)
for the array/configuration contract.

```text
python -m correction.scripts.prepare_phase --reference reference.npz --config phase_config.json --start-frame 0 --stop-frame 120 --output outputs/phase.npz
python -m correction.scripts.run_phase --phase outputs/phase.npz --config phase_config.json --output outputs/corrected_phase
```

A self-contained synthetic contact example is available without a dataset:

```text
python -m pip install -r correction/requirements.txt
python -m correction.scripts.demo --output outputs/contact_example
```

## Source layout

`src/preprocessing` reads retargeted references and constructs interaction
phases; `src/dynamics` holds physics, contact, and replay assessment;
`src/optimization` contains the solvers; `src/pass_traj` writes references and
manifests. `scripts/` contains public entry points. Tests, historical comparison
scripts, datasets, and generated results remain local and are excluded from Git.
