# Policy refinement, October 27 overnight

## Authorization and limits

The user explicitly approved the expanded overnight plan in this chat: "Plan approved. Go ahead with this overnight run." No further approval is needed for implementation, bounded training, diagnosis, rollback, or up to two subsequent focused revisions within this scope.

- Leave the AutoDL instance powered on.
- Do not interrupt the other agent's training or its evaluations.
- Maximum three candidate implementations, six additional GPU wall-hours including evaluation, and two training hours per candidate.
- Stop new GPU work by **2026-10-27 09:30 America/New_York (13:30 UTC)**. Report by **10:00 (14:00 UTC)**.
- One GPU worker at a time; remote processes must enforce their own time limits.
- Preserve the best code/configuration/checkpoint combination. Newest does not mean best.

The local computer and Codex must remain running for scheduled diagnosis and code changes. Heartbeat `v2d-policy-refinement-until-10-a-m` resumes this same chat every 20 minutes until the report deadline. Pause it after the final report. Never put SSH passwords in this repository or logs; the authorized connection is in the user message.

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

## Next action

Finish and review the extension/launcher/evaluator; run CPU checks, commit exact source, deploy separately. Wait until the existing queue (including evaluations) is terminal and GPU processes are gone. Inspect its new results and select the starting policy/state before launching the first bounded diagnostic and training phase. Record every command, result path, PID and budget charge here or in the linked machine-readable ledger.
