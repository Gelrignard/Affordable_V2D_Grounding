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

## Next action for the scheduled continuation

1. Inspect `bootstrap_status.json`, `ledger.json`, live processes and the old queue status. While old work continues, do not launch a GPU job. Do not rerun bootstrap or mutate its deployed source. Stay quiet for unchanged state.
2. If `needs_attention`, read bootstrap/supervisor/training logs and fix the actual error in the isolated local checkout. Commit a new source snapshot; use a new output directory for retry. Count all previous GPU time and training attempts in the same ledger. Failed integration repair within the same method remains candidate1, not a new candidate implementation.
3. If `awaiting_agent_diagnosis`, compare parent and candidate full-rollout `evaluation.json`, `phase_metrics.json`, `rollout_arrays.npz` and `diagnostics.npz`. Inspect first failure, early-phase preservation, handover and final-200 progress, contact traces and training optimizer health. Select/retain best by actual object trajectories. Do not promote solely because reward increased or training terminations fell. `best_checkpoint.json` initially points to the parent and must be explicitly updated only after this review.
4. Continue useful progress or make one focused candidate revision, restoring the preserved best parent when necessary. Subsequent jobs use `experiments/overnight_guard.py` with the **same** remote ledger, old queue path and deadline. Use candidate IDs `candidate1`, `candidate2`, `candidate3` consistently; split phases and resumes count toward the same two-hour candidate cap. New candidate code gets a `codex/` branch and immutable committed source directory. Keep policy and compact state pairs intact.
5. Training CLI: `python experiments/overnight_policy.py train --runtime RUNTIME --checkpoint STATE --output NEW_OUTPUT --phase refine|consolidate --additional-transitions N --max-seconds SECONDS --deadline-utc 2026-10-27T13:30:00Z`. Evaluation CLI: `python experiments/evaluate_overnight_policy.py --runtime RUNTIME --checkpoint POLICY --output NEW_OUTPUT --worlds 16 --tier 1 --episode 5 --variant NAME`. Run both through the guard, with distinct supervisor and worker output directories. Guard arguments: `--ledger ROOT/ledger.json --name UNIQUE --candidate candidate1 --kind training|evaluation --max-seconds N --deadline-utc 2026-10-27T13:30:00Z --wait-queue OLD_QUEUE --wait-seconds 60 --output NEW_SUPERVISION_DIR --cwd COMMITTED_SOURCE -- COMMAND ...`.
6. Check disk before further phases; initial free space was 17 GiB. Do not delete other agents' artifacts. Upload bandwidth was approximately 10 KiB/s; prioritize downloading the best 4.7 MiB policy and small evidence files before the report, while preserving the 153 MiB compact training state remotely if transfer cannot finish in time. Do not start a huge transfer that delays the 10 a.m. report.
7. Stop GPU work by 09:30 NY, report by 10:00 NY, leave the instance powered on, and pause the heartbeat after the report. If the method does not improve, say that directly and retain the best pre-existing policy.
