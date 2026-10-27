# Two additional episodes — October 27, 2026

The user asked to run the successful method on two more episodes after checking the instance. At 13:53 UTC there were no GPU training or evaluation processes. The completed overnight episode5 experiment remains protected and is not resumed.

## Scope and method

Use only the isolated `policy-overnight` checkout, branch `codex/policy-two-episodes-20261027`. Run corrected Tier1 episode16 (spray bottle desk to ground,477frames) then episode18 (sideways trash-can drag,858frames), one GPU worker. Both inputs have one rigid object and43joints, match their portable manifests, and load with the pinned native loader. Their object poses and contact labels equal the original variants. Estimated force guidance is not used.

Neither episode has a compatible native SAC checkpoint. Each therefore starts a fresh native CHORD350M curriculum, retaining SONIC, the native residual policy initialization, seed42 and the original VOC schedule. Use4M GPU replay,2048worlds/batch,JAX memory fraction.60 and23000MiB memory guard. Save the initial step0 policy/state before learning.

Evaluate the completed native349999104 checkpoint on16 uninterrupted full-frame-zero unassisted worlds. Then apply the successful candidate2 refinement for at most60M transitions/30minutes: object-relative contact-phase hand reward, recovery termination,10M object-reward ramp, and the position mixture `(2/3)*exp(-(e/.20)^2)+(1/3)*exp(-(e/.05)^2)`. Preserve observation/action/SONIC/controller/physics/timing semantics and optimizer clocks. Refill fresh replay with the restored policy. Refinement global VOC is zero; random recovery resets may retain native brief assistance. Full-start evaluations have zero reset assistance and no settling or within-rollout reset.

A full native checkpoint is required before refinement: a short or failed bootstrap stops for diagnosis. Evaluate refinement against that same episode's native parent. No automatic promotion. Inspect phase errors, hand/contact traces, optimizer health and all failure tails. A useful result may receive one30M extension within the same budget, after diagnosis. Preserve the better policy and state; never continue a degraded endpoint blindly.

## Bounds and execution

New remote root: `/root/autodl-tmp/v2d/policy_two_episodes_20261027`. It has a separate immutable budget policy and canonical ledger:3training hours per episode,6.5GPU wall-hours total including evaluation, maximum2episode IDs. All jobs use `experiments/overnight_guard.py`, an exclusive GPU worker, wall caps and a21:00UTC calendar cap. Native bootstrap wall cap9000seconds; refinement cap1800seconds, reduced if cumulative remaining budget requires it. Evaluation reserve600seconds per phase, plus60seconds outer supervision reserve around each training child. Leave the instance powered on.

The detached queue validates both input bundles and committed source hashes before its first GPU job. It saves immutable effective configs and parent/input/source hashes, status, PIDs and charged time. It can advance episode18 while the agent diagnoses completed episode16. Existing completed overnight and other-agent artifacts remain intact.

## Preparation evidence

- Local relevant launcher/guard/bootstrap/evaluation/queue tests:59passed using the existing `.soma-env` Python. Plain Python3.12 lacks NumPy/PyArrow; that interpreter's two import errors were dependency availability, not test failures.
- Native runtime pinned at `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.
- Episode18 archive SHA256 `265b559c253e2fe55ffd578b61039a30ad6ce0ea19791e20edec4a8f1f0ce12a`; verify after upload before extraction.
- Episode16 uses the existing remote portable bundle; preparation verifies all job file hashes.
- Status before launch: preparation and input transfer only. Append committed source, launch PIDs and startup verification once available.
