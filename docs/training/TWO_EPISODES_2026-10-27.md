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

- Startup verified14:31:45UTC: queue532522, native learner532561; episode16 at2.9696M transitions,51.924k transitions/s, finite recent metrics, actor loss-1.7684, critic loss2.0944, zero current-window skipped updates. Initial native step0 pair was saved with all learning clocks/replay zero and seed42. GlobalVOC1 is the intended first native bootstrap stage; refinement will use globalVOC0. GPU21612MiB, below23000MiB guard; oneGPUworker. The actual learner is progressing, not just a queued process.
- Current plan and both effective native configs downloaded and SHA-verified in `outputs/two_episode_deployment/`; launch and CPU-test evidence are preserved there. Remote `startup_verification.json` records the checks above. Let the bounded queue proceed; judge quality only after the full unassisted parent/refinement evaluations. Stay quiet on unchanged heartbeats.

### 12:42 New York — episode16 parent evaluated; refinement running

Native bootstrap completed349999104 transitions in7627.640chargedseconds (2h7m8s), followed by40.020seconds evaluation. The full477-frame16-world unassisted parent has AUC.282167, MPPE61.62495cm, SP-SR0, MP-SR0. No non-finite worlds or evaluation assistance. Early-third MPPE3.072cm/AUC.773364, middle74.326cm, late107.476cm; major failure begins after early acquisition. Diagnose actual hand/body/object timing before attributing a cause.

The same queue automatically began episode16 precision refinement16:35:40UTC, learner567424, restoring the exact native349999104 parent and policy-filled fresh replay. At16:42:34UTC14.6944M additional transitions,49.851k/s, finite actor/critic losses and zero latest-window skips, globalVOC0; GPU21612MiB. Source/plan and budget are unchanged. Let this bounded screen finish and evaluate; episode18 remains queued. Native evaluation evidence is being downloaded for CPU diagnosis. Do not start another GPU worker or resume any historical source.

- Parent evidence downloaded and SHA-verified: `outputs/two_episode_analysis/episode16_native_analysis_v2.tar.gz`, SHA4da2a3ffe7eda01e60b6b84ad052be072804483a7a47f3c79e8bd1b286f25dd6; full arrays, contacts, native logs and manifests extracted under `episode16_native/`.
- CPU optimizer audit (`episode16_native_optimizer_audit.json` and reusable.py): all6836records finite; actor/critic mean skipped fractions.01699%/.05302%. Failures rise at global-VOC removal and introduction of immediate starts: adjacent5M reference-end fractions50.65%→31.65% at210M; palm terminations27.56%→45.70% at25% immediate starts,47.50%→64.16% at50% starts. These are window averages and temporal evidence, not isolated causality. Final training object error3.28cm is measured on reset-interrupted/recovery-start samples, whereas the full rollout MPPE is61.62cm. Do not confuse them. No numerical evidence supports changing the optimizer/replay/precision configuration now.
- At16:46:50UTC the refinement has26.5728M additional transitions,49.604k/s, all logged metrics finite, zero latest-window skips, globalVOC0, immediate-start probability.5, and21612MiB GPU. Continue this bounded screen; no additional GPU job or hyperparameter revision was launched.

- Full-rate parent failure audit (`episode16_native_failure_diagnosis.json`, reusable `episode16_native_failure_analysis.py`): sustained object-position error>10cm (10frames) precedes native pelvis failure(.7m or1.5rad) in all16worlds by21–98frames, median50.5frames. At object failure, pelvis error remains.078–.322m/.169–1.225rad, below those native limits. Initial body drift recovers byframe95 beforeleftcontact. Later subthreshold body drift can contribute (sustained>.1m beforeobjectfailure in14/16); do not infer independent causality. Object failure precedes sustainedpalm failure in9/16; palmtracking leads in7/16. Worlds9and10 show clear contact-loss examples beforelargebodyfailure, while others retain contact during drift. Therefore the large laterbody deviations do not establish bodycollapse as the initiatingcause. No optimizer or methodchange is justified before the running refinement evaluation.
