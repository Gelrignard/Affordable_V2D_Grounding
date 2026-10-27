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

## Deployment and preflight correction

Source2f6db5396841e3ff76720a4e6787fe3182a00e20 was committed and SHA-verified remotely; all79 affected tests passed in the pinned cloud environment with CUDA hidden. Both episode manifests and the episode18 archive match their local hashes; native typed configuration preparation passed. The initial plan SHA is075274d164ea426b221a0311f5640f5c376e3ed232012fa4d25e7d276330f503.

The first queue supervisor532337 started14:24:13UTC but stopped during preflight before any GPU allocation or ledger creation: PyArrow read_table interpreted the Hive-shaped runtime directory as a dataset and tried to merge a dictionary partition key with a string column. Corrected verification to use single-file ParquetFile reads, matching the pinned native loader. The focused input regression now uses the actual sequence_id=.../robot_name=g1 directory shape; all6queue tests passed. Preserve failed launch/status/log and original plan, then deploy the committed correction and a new immutable plan within the same trial root and unchanged budget.

### 10:27 New York — corrected queue launched

Committed execution source3560a4bc868f3b56891b5946516e6714fd493ced; source archive SHA2568bacd0080f018e25ca02dedb391306a3e9e56702e1dd16d9879f3e98a9871042 verified before extraction. Both actual remote runtime inputs and configs passed the corrected CPU preflight;6focused tests passed on the deployed revision. Earlier79nativeCPUtests cover unchanged launcher/refinement/evaluation code.

The initial CPU-only failed launch is preserved at remote `attempts/preflight_001/`; the original plan remains intact. Current immutable plan is `plan_retry1.json`, SHA256bcee3ee8f0255cef5d5d5d0afe8a72c8e974dc72154ec2f897b9b4021ef8aa36. Input/config paths and all budget limits are unchanged. No GPU time or ledger was consumed by the failed preflight.

Detached queue supervisor532522 launched14:27:46UTC after verifying GPU idle. Inspect `queue_status.json`, `ledger.json` and the active episode status; never duplicate it. Episode16 runs first, episode18 second. The existing heartbeat is reactivated for this newly requested trial, with the final GPU deadline21:00UTC.
