# Policy refinement, October 27 overnight

## Authorization and limits

The user explicitly approved the expanded overnight plan in this chat: "Plan approved. Go ahead with this overnight run." No further approval is needed for implementation, bounded training, diagnosis, rollback, or up to two subsequent focused revisions within this scope.

- Leave the AutoDL instance powered on.
- Do not interrupt the other agent's training or its evaluations.
- Maximum three candidate implementations, six additional GPU wall-hours including evaluation, and two training hours per candidate.
- Stop new GPU work by **2026-10-27 09:30 America/New_York (13:30 UTC)**. Report by **10:00 (14:00 UTC)**.
- One GPU worker at a time; remote processes must enforce their own time limits.
- Preserve the best code/configuration/checkpoint combination. Newest does not mean best.

The local computer and Codex must remain running for scheduled diagnosis and code changes. Heartbeat `v2d-policy-refinement-until-10-a-m` resumes this same chat every 10 minutes until the report deadline. Pause it after the final report. Never put SSH passwords in this repository or logs; the authorized connection is in the user message.

## Workspace and cloud paths

- Isolated checkout: `C:/Users/11534/.codex/worktrees/policy-overnight/v2dchallenge`
- Initial branch: `codex/policy-overnight-20261027`
- Initial commit: `a01610201b1f24a9071c030a8c19653be33fbb2c`
- Other agent's local checkout, read-only: `C:/Users/11534/OneDrive/Desktop/v2dchallenge/.worktrees/tier2-rl`
- Native runtime source for local inspection: that checkout's `vendor/flash-runtime`.
- Cloud Python: `/root/autodl-tmp/v2d/native_chord_env/bin/python`
- Cloud runtime: `/root/autodl-tmp/v2d/project/vendor/flash-runtime`
- Job: `/root/autodl-tmp/v2d/project/outputs/training_jobs_v1/tier1/episode_000005/corrected`
- Existing active queue: `/root/autodl-tmp/v2d/flash_sac_host_10m_20261027/full_queue`
- Completed 5M run: `/root/autodl-tmp/v2d/flash_sac_5m_20261026/full_queue/native_5m`
- New experiment root (reserved): `/root/autodl-tmp/v2d/overnight_policy_20261027`
- Root-workspace CPU evidence: `C:/Users/11534/OneDrive/Desktop/v2dchallenge/outputs/policy_audit_20261026`

Do not modify the existing active queue, native runtime, prepared job, other branch, or old checkpoints. Deploy committed source to a separate immutable directory under the new experiment root. Record the Git commit and file hashes before each launch. Keep code and model restoration paired; checkpoint replay is not saved and any fresh replay refill must be recorded.

## Method and evidence

The first candidate warm-starts the strongest evaluated native FlashSAC policy, using fresh 5M GPU replay, 2048 worlds and batch 2048. Preserve network observations/actions, SONIC, physics, and object-reference timing. Use object-relative hand guidance during reference contact, smooth contact transition weights, recovery-aware training termination, and gradual object-pose reward emphasis while preserving lower-body stabilization. No world model, estimated-force target, or online trajectory optimizer. Do not restart full virtual-object assistance; explicitly remove reset assistance for final full-start consolidation.

The existing 5M policy is the initial fallback: AUC 0.3348859671, MPPE 17.38267926 cm, SP-SR/MP-SR 0. Sixteen worlds, one trained seed. Fourteen worlds first sustain >10 cm position error at frames 748-751, immediately following corrected palm jumps at 744-746. This motivates tracing actual contacts and motion, not assuming that smoothing has already solved the issue. The old Dex-inspired PPO remained at assistance 1.0 and is not the initial warm-start source.

Screen after 20-30 minutes. Extend up to 45-60 minutes if physical trajectory errors are improving but aggregate scores are noisy. Stop numerical corruption immediately. Stop sustained clear regression and restore the parent. Continue useful progress up to the candidate's two-hour cap. Evaluate unchanged full-start unassisted object trajectories, not modified training termination counts or reward alone.

After each candidate, diagnose the first remaining failure and change one focused mechanism: a transition if failure remains localized; recovery practice if failure moves later; reward/optimizer calibration if prior behavior collapses. Do not run a parameter sweep or a new baseline campaign. No algorithm switch merely because the score is flat.

## Live log

### 00:47-00:50 New York — approved, preparing

- Activated the same-chat heartbeat and marked the local plan approved.
- Existing queue is still running; at the read-only snapshot it had 228,966,400 transitions, VOC 0.0, approximately 22.9k transitions/s. Its processes are 330497 (queue), 330500 (supervisor), 330501 (training); confirm current PIDs instead of assuming they remain valid.
- Free cloud disk at that snapshot: approximately 17 GiB. Do not delete other agents' checkpoints. Retain starting/best/final and bounded intermediate checkpoints for our own runs.
- Parallel work: `local_audit` implements the opt-in runtime extension; `official_baseline` implements the bounded warm-start launcher; `training_literature` implements full-rollout evaluation/diagnostics. All edits belong to this isolated checkout. Root owns supervision, deployment, integration and decision-making.
- No new GPU job has started.

### 01:09 New York — committed, verified, cloud starter armed

- Candidate code commit: `0ba12c7bab8d69cf69e2d24625575b1006fbfcac` on `codex/policy-overnight-20261027`. Initial implementation and hardening are recoverable as `ac2c779` and `a32b64e`.
- Deployed immutable source: `/root/autodl-tmp/v2d/overnight_policy_20261027/sources/0ba12c7bab8d69cf69e2d24625575b1006fbfcac`.
- Deployment archive SHA256: `2409bb6eaa40360f96883cd80ed3a8fb5816d13b2b701c4a74ac2109038276ef`; verified before extraction. Original runtime and prepared input remain untouched.
- Detached bootstrap PID **424694**, started in its own process session. **Do not launch another bootstrap.** Full command is preserved remotely in `bootstrap_launch.json`.
- The existing GPU training PID was still 330501 at the last check, using 8262 MiB. New bootstrap is CPU-only while waiting for the entire existing queue. It requires terminal queue status, dead queue supervisor and an idle GPU; each guarded launch checks again.
- Remote `cpu_verification_final.log`: all **38 tests passed**, including actual Warp CPU kernels and native config/reset integration. After the last warmup/config-manifest changes, all **13 affected launcher/selection tests passed** again (`cpu_verification_launch.log`). Actual saved state 300011520 successfully instantiated the native objective, termination, reset, curriculum and TrainingConfig on CPU. GPU integration remains pending the old queue's completion.
- Important implementation details: raw replay has 13 objective terms; optimizer horizon remains 683592; actor/critic/normalizer/optimizer restore from compact state; replay is fresh 5M, filled using the **restored stochastic policy**, not uniform random actions. Exact actor parity is checked before updates. Reward-ramp origin persists across resumes. Global VOC is zero, random recovery reset stabilizer remains 1 with 50-step freeze; final consolidation uses exact frame zero, reset VOC zero and freeze zero.
- The best currently resumable 5M checkpoint by AUC is **300011520**, AUC **0.3421999626**, MPPE **18.02266657 cm**, both success scores zero. The earlier detailed phase diagnosis concerned the later 349999104 checkpoint (AUC 0.3348859671); keep that distinction. Bootstrap will also consider the pending 10M queue's evaluations before choosing a parent.
- Bootstrap sequence: choose highest evaluated AUC among available matching policy/state pairs; reproduce the parent with the new unassisted evaluator; write effective config and parent/input hashes; run candidate1 refinement for at most 1800 seconds / 60M additional transitions; evaluate its new checkpoint; stop at `awaiting_agent_diagnosis`. The outer guard charges training up to 1860 seconds including process overhead, and each evaluation up to 600 seconds, against the common six-hour ledger.
- Remote status/outputs under the experiment root: `bootstrap_status.json`, `bootstrap.log`, `ledger.json`, `starting_checkpoint.json`, `best_checkpoint.json`, `candidate1_screen_manifest.json`, `candidate1_screen_planned_config.json`, `parent_evaluation/`, `candidate1_screen/`, `candidate1_screen_evaluation/`, `first_screen_comparison.json`, and `supervision/`. Some are created only after the waiting stage. A missing ledger while waiting means no new GPU work has started.
- Graceful and forced stopping cover the entire training process group. All jobs independently obey the calendar deadline. Shared ledger lock prevents duplicate workers; a stale running ledger entry must be reconciled against live PIDs, never simply erased.

### 02:30 New York — original queue complete, first refinement starting

- Original 10M host-replay run completed 349999104 transitions at 06:17:17 UTC after 15299.75 seconds; its full evaluation queue completed around 06:22 UTC. Final policy is strongest: ADD-AUC **0.4248971834**, MPPE **14.46966510 cm**, both success scores zero. Its 300M checkpoint scored 0.372207596 / 18.71609198 cm. All seven queued evaluations completed.
- Selected parent state/policy: `/root/autodl-tmp/v2d/flash_sac_host_10m_20261027/full_queue/native_host_10m/checkpoints/{state,policy}_349999104.safetensors`. Native compact host-replay states are compatible with fresh GPU replay; host replay is installed only in the old process, not encoded in the checkpoint.
- New full-rollout parent evaluation completed, consuming **45.015 seconds** in the common ledger. Measured baseline is **AUC 0.4220892011 / MPPE 14.55741313 cm**, success scores zero. The original bootstrap stopped before training because its historical-score equality tolerance was too tight (1e-4). This was a harness issue, not a training-method result.
- CPU comparison verified identical checkpoint SHA, mesh samples, frame sequence, cohort, source parquet path, explicit reset and simulator rates. Reference discrepancies were float32 storage rounding (<6e-8); initial object pose difference <8e-7, first100-frame max position difference 5.18e-6 m, with later physical divergence. Exact GPU repeatability was not established; do not present that explanation as proven. `parent_repeat_diagnosis.json` retains the evidence.
- Fixed bootstrap in commit **5bcae1c972573b7c5c0c9d69337347a97d7d3206**: require identical evaluation contract and finite results; allow absolute repeat drift up to .01 AUC / .5 cm MPPE with unchanged success scores; retain both measurements and compare candidate against the new evaluator's parent. Added safe resume of a completed parent check only when no training/running ledger job exists. All 10 relevant tests passed locally and remotely. No extra GPU evaluation was repeated.
- New immutable source: `ROOT/sources/5bcae1c972573b7c5c0c9d69337347a97d7d3206`; archive SHA256 `0bff7bb5091a81d3cc20dbb5f8bf268283bb92040f53b61d697963b17436a1f2` verified before extraction. Original source snapshot is preserved.
- Recovery bootstrap PID **465551**, native training PID **465560**, started **06:30:24 UTC**. **Do not duplicate this worker.** `bootstrap_recovery_launch.json` records the exact command with `--resume-parent-evaluation`; old failed state is preserved as `bootstrap_previous_*.json`. Current status is `training_first_screen`. Native scene initialization is underway; actor parity/first optimizer updates remain to be checked.
- Parent first object-threshold violations occur at frames **746–750** across all16 worlds, confirming the handover remains the primary target. Parent handover730–850 MPPE **21.366 cm**, final200 MPPE **53.487 cm**. Full parent traces are in `parent_evaluation/`.

### 02:38–02:45 New York — startup repairs and memory adjustment

- First training attempt stopped after 60.095 seconds before learning: Python 3.12 runtime protocol checks require explicit wrapper properties. Fixed and committed in `7cf933f6912ff73a275e30ec19996ba77395e882`; 26 affected native CPU tests passed. This is an integration repair within candidate1.
- Retry1 (bootstrap PID466027, learner PID466036) passed exact restored actor parameter/action parity and native SAC compilation. The memory guard stopped it at **23885 MiB** after **125.044 seconds**. It collected only2048 transitions into fresh replay; the saved global update count remains682772. Keep this failed attempt and its checkpoint in `candidate1_screen_retry1/`.
- Total charged time before retry2 is **230.153 seconds**: parent evaluation45.015, first failed startup60.095, second failed startup125.044. Both workers have exited; GPU was idle at inspection. The original10M queue remains complete.
- Native source inspection found one replay allocation. The previous5M run also reached about23887MiB under its higher guard, and an older4M startup reached that allocator high-water mark. Replay reduction alone is insufficient to guarantee headroom with the existing allocator settings.
- Adjust candidate1 to **4M fresh GPU replay** and outer JAX memory fraction **.75** (from.84), preallocation false; keep the **23000MiB guard**. Ring storage saves about2146MiB. This is a resource adjustment, not evidence of training improvement; actual GPU initialization and learning must pass before leaving it unattended.
- Retry2 must use a new immutable source and output `candidate1_screen_retry2`, the same ledger and candidate ID. Restart from the original evaluated parent only after verifying the failed warmup checkpoint contains no learned updates. Preserve its evidence. No second parent GPU evaluation is needed.
- Read-only checkpoint byte comparison found only reward-normalizer, exploration, RNG and environment-step leaves changed during that single warmup collection; all actor/critic/target/temperature parameters and optimizer states were unchanged. Parent and stopped checkpoint both report682772 updates. Disk available16.176GiB.
- Audited the restored optimizer schedule: critic learning rate remains at its1.5e-4 floor; actor is about2.2514e-4 and would decay to about2.0532e-4 over60M additional transitions. Retain clocks, moments and horizon; this screen is not using a vanishing learning rate.

### 02:50–02:57 New York — memory setting did not resolve startup

- Retry2 source commit `5140d106ace195e085d2f5b35ea3f6e03d669b14`, archive SHA256 `67c47d84e8681d9c0df06db55b430f104096f75d7370992cb0429744e712f0f2`. All29 relevant CPU tests passed locally and remotely. Bootstrap PID469320, learner469331; both now exited.
- The startup recovery gate verified177 protected state tensors and all deployable actor tensors unchanged in retry1. Its report is `candidate1_screen_retry2_failed_startup_verification.json`.
- Retry2 again hit the memory guard at **23883MiB**, despite provenance confirming4M replay and JAX fraction.75. It stopped cleanly after121.107 seconds with38912 fresh warmup transitions, below409600 replay minimum. No training result exists yet; do not call the method successful or report it as running.
- Read-only installed JAX0.10.2 parser confirms the fraction was accepted; no competing modern fraction, unified memory, allocator or XLA flag overrides were present. Native source contains no fraction override. Both attempts grow to about18815MiB before compilation, then23883MiB afterward. A repeated memory bucket is an explanation to investigate, not yet a confirmed cause. Add allocator telemetry before any further resource change.
- Downloaded and hash-verified the original best parent policy AND153MiB compact state into this isolated checkout's `outputs/overnight_best/`. PolicySHA `1ff07c00361ffbeb7edc41f0aba23dbebc9eadc7c74aa40fa7e0cff6fdfc8722`; stateSHA `1eae67b3272adf75eba9ecfbbb6f21a22bd1b8d76db8df1f6d44cb820e21fd07`. Preserve both as the rollback pair.

### 03:00 New York — allocator diagnosis measured

- Bounded single-collection diagnostic `candidate1_memory_diagnosis`, guardPID472458/learner472466, source `ec709b593c1c2e376e603f145a6e3b3c94524380`. Common ledger charges this startup as `smoke`; it collected only2048 transitions and performed no learning. Its complete launch/config/hash manifest and staged telemetry are preserved remotely.
- `memory_stages.jsonl` shows JAX pool **17465081856 bytes** with only **9247339520 bytes live** after compilation. JAX limit was18947768320bytes, confirming.75 was honored. Warp pool use remained1390348506bytes. First environment step raises totalGPU use18835→23871MiB while both tracked pools stay unchanged. That additional allocation's owner is not yet established; do not label it a JAX allocation or claim a proven ONNX cause.
- Lower outer JAX fraction to **.60** to clip its oversized pool to about15158214656bytes, preserving more than5GB beyond observed JAX live usage while reclaiming about2.15GiB of reservation. Keep4M replay and23000MiB total guard. This follows measured allocation data; optimizer-update peak and sustained GPU behavior still need validation.
- Next full screen is `candidate1_screen_retry3` from the same evaluated parent, using a new immutable source commit and the same ledger. All prior training attempts were below the fresh replay warmup threshold and did not change learned weights.

### 03:03–03:07 New York — first refinement learning successfully

- Deployed source commit **2951d24a059ed9840f6818f3b9e1884ee22d3ab4**, archive SHA256 `a65522fd0f1337af8bf4130ea373ff791ab792453a7c9a86d3822b9daae60061`; verified before extraction into its immutable source directory.
- **Active bootstrap PID475486, learner PID475497**, output **`candidate1_screen_retry3`**, started07:02:52UTC. **Do not duplicate or stop this working screen.** Exact invocation is `bootstrap_retry3_launch.json`; current bootstrap state is `training_first_screen`.
- At07:05:44UTC the screen had **2508800 additional transitions**, replay2471936, finite actor loss-1.79856 and critic loss2.04824, **zero skipped actor/critic updates**, throughput43906 transitions/s, globalVOC0. This verifies learning beyond fresh-replay warmup. Observed GPU peak **21687MiB** matches the measured reservation diagnosis and is below the unchanged23000 guard. No quality improvement has been established yet.
- Pre-screen total charged GPU time including parent evaluation, failed startups and diagnostic was **475.508seconds**. Continue using the same ledger; previous attempts are retained. Current screen is capped at60M additional transitions/1800seconds and will automatically run full unassisted evaluation, then stop at `awaiting_agent_diagnosis`. Expected first comparison around07:30UTC/03:30NY if throughput holds.
- Updated the existing heartbeat prompt to reflect the active worker,4M replay and.60 allocator setting; it remainsACTIVE every10minutes until14:00UTC. Do not create another automation. Local fallback policy/state pair is hash-verified in `outputs/overnight_best/`, with `local_backup_manifest.json` and parent evaluation JSONs.

### 03:32–03:40 New York — first screen improves physical rollouts

- Screen1 completed59998208 additional transitions in1570.659seconds charged training time, followed by50.028seconds evaluation. Total ledger usage is **2096.195seconds (0.5823 GPU hours)** including all startup work. No worker remains active at this decision point; bootstrap is `awaiting_agent_diagnosis`.
- Full frame-zero unassisted evaluation,16worlds/1003frames, no within-attempt resets: **ADD-AUC .422089→.573927**, **MPPE14.557→5.976cm**, **SP-SR0→.8125**, **MP-SR0→.125**. Same reference parquet, timing, cohort, zero global/resetVOC and zero settling. This is actual object-rollout improvement, not a reward/termination proxy.
- Early acquisition0–399 improves MPPE3.79→2.81cm and AUC.575→.717. Aggregate0–729 MPPE is essentially unchanged3.820→3.808cm despite two new middle failures. Handover730–850 MPPE21.366→7.521cm; final20053.487→13.899cm. Persistent failures: world14 near441, world6 near643, world0 near752; world8 drifts after802. Worlds9/10 briefly exceed10cm then recover, so do not tighten termination and remove that recovery opportunity.
- Contact samples show world0 loses right-hand contact around745, then both by760; world8 lacks sampled contacts765–790. World14 has a108N sampled left contact near440 before displacement at445. Sparse final-physics-substep force samples do not establish the complete contact impulse or causality. All detailed rollouts/diagnostics are downloaded under `outputs/overnight_analysis/{parent,screen1}`.
- Full optimizer audit: no non-finite metrics in1172records. Unlike the early spot checks, the complete run has isolated adaptive mixed-precision skips:47/58182actor(.081%) and35/116364critic(.030%), with no consecutive skipped logging windows or loss-scale collapse. Loss scale stays2^22–2^24 and ends2^23; late entropy45.687 matches target45.677. No optimizer/algorithm change is justified.
- Promote screen1 checkpoint **409997312** as the current best while preserving the original. PolicySHA `9d8815f0bba03562621297c991dc87ad8e46e25814f4adf3e4ebc3bd203638dc`; stateSHA `92cb3fd5b287218ee51cbb7c317a9933d2e9c27aff30ed67fc235cf9c388a532`.
- Next action: **candidate1_consolidation1**, at most60M additional transitions/1800seconds, from screen1. Use existing consolidation phase: all resets exactly framezero, resetVOC0, freeze0, shoulder spread0; retain learned weights, optimizer clocks, reward ramp origin349999104, final reward weights and recovery termination. This targets full-trajectory repeatability after successful handover. Another refinement block was plausible because optimization is healthy; choose consolidation because most worlds already traverse handover and late training progress has slowed. Evaluate before extending; retain screen1 if consolidation regresses.
- New `experiments/overnight_phase.py` runs exactly one manifested train/eval phase through the same guard/ledger and stops at its own `<name>_status.json` awaiting diagnosis. It never promotes automatically. Check that status as well as the completed bootstrap in future heartbeats.

## Next action for the scheduled continuation

1. Inspect `bootstrap_status.json`, `ledger.json`, live processes and the old queue status. While old work continues, do not launch a GPU job. Do not rerun bootstrap or mutate its deployed source. Stay quiet for unchanged state.
2. If `needs_attention`, read bootstrap/supervisor/training logs and fix the actual error in the isolated local checkout. Commit a new source snapshot; use a new output directory for retry. Count all previous GPU time and training attempts in the same ledger. Failed integration repair within the same method remains candidate1, not a new candidate implementation.
3. If `awaiting_agent_diagnosis`, compare parent and candidate full-rollout `evaluation.json`, `phase_metrics.json`, `rollout_arrays.npz` and `diagnostics.npz`. Inspect first failure, early-phase preservation, handover and final-200 progress, contact traces and training optimizer health. Select/retain best by actual object trajectories. Do not promote solely because reward increased or training terminations fell. `best_checkpoint.json` initially points to the parent and must be explicitly updated only after this review.
4. Continue useful progress or make one focused candidate revision, restoring the preserved best parent when necessary. Subsequent jobs use `experiments/overnight_guard.py` with the **same** remote ledger, old queue path and deadline. Use candidate IDs `candidate1`, `candidate2`, `candidate3` consistently; split phases and resumes count toward the same two-hour candidate cap. New candidate code gets a `codex/` branch and immutable committed source directory. Keep policy and compact state pairs intact.
5. Training CLI: `python experiments/overnight_policy.py train --runtime RUNTIME --checkpoint STATE --output NEW_OUTPUT --phase refine|consolidate --additional-transitions N --max-seconds SECONDS --deadline-utc 2026-10-27T13:30:00Z`. Evaluation CLI: `python experiments/evaluate_overnight_policy.py --runtime RUNTIME --checkpoint POLICY --output NEW_OUTPUT --worlds 16 --tier 1 --episode 5 --variant NAME`. Run both through the guard, with distinct supervisor and worker output directories. Guard arguments: `--ledger ROOT/ledger.json --name UNIQUE --candidate candidate1 --kind training|evaluation --max-seconds N --deadline-utc 2026-10-27T13:30:00Z --wait-queue OLD_QUEUE --wait-seconds 60 --output NEW_SUPERVISION_DIR --cwd COMMITTED_SOURCE -- COMMAND ...`.
6. Check disk before further phases; initial free space was 17 GiB. Do not delete other agents' artifacts. Upload bandwidth was approximately 10 KiB/s; prioritize downloading the best 4.7 MiB policy and small evidence files before the report, while preserving the 153 MiB compact training state remotely if transfer cannot finish in time. Do not start a huge transfer that delays the 10 a.m. report.
7. Stop GPU work by 09:30 NY, report by 10:00 NY, leave the instance powered on, and pause the heartbeat after the report. If the method does not improve, say that directly and retain the best pre-existing policy.
