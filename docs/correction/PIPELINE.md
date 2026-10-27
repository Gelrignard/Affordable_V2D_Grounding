# Dynamics correction and contact-orientation grounding

The input is an episode's human motion, object pose observations, contact evidence, object geometry, and robot model. These stages operate on motion artifacts.

## Contact-orientation grounding

The retargeter maps human motion to G1/Dex3 with explicit joint names and coordinate conventions. Contact evidence supplies object-relative regions and hand choices. Robot collision surfaces, palm orientation, joint limits, and reachability constrain candidate contacts.

```text
python retargeting/scripts/run.py --input human_motion.npz --method contact_aware --output outputs/episode/retargeted
```

For released observations, use `data_tools.fetch_sample` and `data_tools.decode_tier1` with the intended `--tier` and `--episode`. The decoder's name is historical; its explicit Tier 2 path retains visibility and imputation masks. See [SOMA setup](../SOMA_SETUP.md).

## Primary dynamics correction

The correction core fits contact patches, allocates feasible forces, and refines robot commands. Force allocation respects friction, normal-force, and nominal actuator limits. Spline refinement preserves contact geometry while regularizing joint motion. Stage status and physical rollout status are recorded separately.

The generic phase interface accepts geometric reference arrays and explicit mass, inertia, friction, and contact-patch assumptions. It smooths the object reference before estimating required wrench and checks dense frames after the knot-level solve. See the [reference interface](INTEGRATION.md).

The full-episode adapter additionally infers interaction phases and creates a portable policy reference. Its staged force profile is quasistatic; simulation determines the actual object motion. Neither a small optimization residual nor a generated reference certifies successful physical execution.

```text
python -m correction.scripts.correct_rl_pilot_episode --input outputs/episode --output outputs/episode/correction
```

The input directory contains `human_motion.npz` and `retargeted/contact_aware.npz`. The `rl_pilot` module names and `v2d_rl_pilot_v1` schema are retained for artifact compatibility. Export and validation commands are documented in the [artifact format](RL_PILOT_ARTIFACT_FORMAT.md).

## Policy grounding

The [policy pipeline](../training/POLICY_HANDOFF.md) consumes prepared original/corrected references and contact guidance. Training simulates whole-body motion and physical contacts. Evaluation starts at frame zero without object assistance or within-attempt resets and retains every world and frame.

Observed targets, planning references, simulated outcomes, and evaluation references are distinct artifacts. Unobserved or interpolated frames remain marked. Clean reference data never becomes a Tier 2 training target.
