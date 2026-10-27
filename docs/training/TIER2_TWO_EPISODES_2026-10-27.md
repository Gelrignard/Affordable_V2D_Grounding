# Two Tier 2 episodes — October 27, 2026

The user requested two Tier 2 episodes using the successful refinement method, without spending GPU time on a separate CHORD comparator. They also requested checks every 30 minutes, delegated to a monitoring subagent, with detailed checks kept out of the main conversation.

Work only in the isolated policy-overnight checkout on codex/policy-two-episodes-20261027. The completed episode 5 and Tier 1 episode 16/18 trials remain protected. This is a new authorized two-episode trial, with its own immutable plan and ledger; it does not extend either earlier trial's limits. Do not start the 90-episode run under this authorization.

## Scope and method

- Tier 2 episode 3: alternating mug handle lifts, 781 frames.
- Tier 2 episode 25: repeated bottle rotations and handovers, 859 frames.
- Corrected portable jobs are taken from the existing tier2-rl checkout, with all job/source/asset hashes verified before copying. Tier 2 supplies noisy motion and contact inputs; it shares the underlying scenes with Tier 1. No clean Tier 1 reference enters training.
- No compatible same-episode native FlashSAC parent was found locally or on the instance. Each episode therefore has one training lineage: fresh native seed42/SONIC initialization and native curriculum to349999104 transitions, then the same candidate2 precision refinement for at most60M further transitions. The initial training is needed to acquire the skill; it is not a separate comparison experiment. Bootstrap evaluation is explicitly disabled for this trial.
- Refinement retains native FlashSAC, 4M GPU replay, 2048 worlds/batch, memory fraction.60 and23000MiB guard. Restore policy/optimizer/normalizer clocks exactly and refill fresh replay with the restored policy. Keep contact-phase object-relative hand guidance, recovery termination, original10M reward-ramp origin, and position shaping2/3 native.20m exponential plus1/3 .05m exponential. No parameter sweep or method change.
- Refinement global VOC is zero. Recovery resets may retain brief native stabilization; all evaluation is exact frame0, zero global/reset assistance, no settling and no within-attempt resets. Every frame and world, including failures, must be retained.

## Execution contract

Remote root: /root/autodl-tmp/v2d/policy_tier2_two_20261027.

Use the v2 two_episode_queue plan, candidate IDs tier2_episode_000003 and tier2_episode_000025, and exactly one guarded GPU worker. Do not restart a queue that already has status/ledger artifacts. Inspect plan.json, queue_status.json, ledger.json, active status/log/metrics and live PIDs before any action. Verify that the earlier episode16 continuation has exited before launch. Keep guard wait_queue pointing to the completed old10M queue, not an awaiting-agent-diagnosis trial root.

Immutable limits: total23400 GPU wall seconds including evaluation; training10800 seconds per episode; maximum2 candidate IDs. Native acquisition has9000s child cap, refinement1800s child cap; supervision time counts toward the same caps. Every job enforces its own wall and calendar limits. Hard GPU deadline2026-10-28T03:00:00Z. Leave the instance powered on. No additional training extension is queued; evaluate useful saved refinement checkpoints before considering any further expenditure.

Commit code before launch, archive that exact source, verify deployment SHA256, preserve source/input/effective configuration/checkpoint hashes and parentage, and record PIDs and charged time. Preserve native and selected refined policy/state separately. No automatic promotion.

## Evaluation and reporting

Run one full unassisted16-world evaluation after refinement per episode. The evaluator's input-reference scores are diagnostic because Tier 2 inputs are noisy. Recompute against the clean released reference on CPU using verified official conversion/metric code, without fitting achieved trajectories to the target or using the clean target in training. Record conversion and dataset provenance, frame coverage and remaining failure tails. Published CHORD results may be cited as a published reference; do not present a two-episode result as directly comparable to a full-tier aggregate or invent per-episode comparator values.

Judge uninterrupted object trajectories, early/middle/late error, actual/reference hands, contacts, pelvis failures and optimizer health. Reward or fewer training terminations is not evidence of task success. Keep raw inspection logs and arrays in ignored outputs/tier2_two_episode_analysis and outputs/tier2_two_episode_monitor. Report only meaningful results, failures needing the user, or completion in the main conversation.

## Monitoring

The existing heartbeat is changed to every30minutes. Delegate routine inspection to /root/local_audit (or a replacement monitoring subagent if unavailable). Return a compact state-change summary to the parent; do not copy routine metric logs into the main context. The scheduler still wakes this chat; delegation limits the detailed material carried back. Pause the heartbeat after the final Tier 2 report, not merely after the earlier Tier 1 trial completes.

## Launch record

Committed execution source54e2a7ceef0e7558208e3eb4e77950aa7d60ffaa. Deployment archive SHA256dbdd68d6a46c4866ed740b64e18530bf37721bd7ebcd5d1ac8875996019d8f2e; portable input archive SHA25623e654e90976b23e5826bf28868c0a732677bd3c131ab4fddde0f4505d5529e2. Both archives were verified before extraction. Both actual Tier2 inputs/configs passed CPU-only preflight, and91 affected pinned-native CPU tests passed. CUDA was hidden during preparation/tests.

Immutable remote plan.json SHA2560978985aa8d55b62f84399defeec8bc0bc894687928f4cb7bf6460bb5f9576d9. Local verified plan/config copies, preparation/test logs and launch evidence are under outputs/tier2_two_episode_deployment. Native config hashes: episode3 b8121f8b905a827a109b1e56a613191bbe939fb13bb90842a8f3c4008a1c87ca; episode25 90edc789153317282d9ce18ceb499e6e3e2e965ad28f3fee96b113c7d53ec944.

Detached supervisor633626 launched2026-10-27T20:15:47.565332Z. Earlier Tier1 episode16 continuation was complete, all its phase PIDs were dead, GPU compute processes were absent, and6.9GB disk space was available before this launch. The new queue runs episode3 then25; inspect queue_status.json and live PIDs to establish its current state. Never duplicate or restart it. The heartbeat now checks every30minutes through03:30UTC; it must pause after the final report. Startup learner progress verification follows separately.

Independent startup passed20:19:19UTC: queue633626, guard633659, wrapper633666, learner633667. Episode3 reached1,740,800 transitions at53.6k/s; all34 metric records finite, zero latest actor/critic update skips, native initial globalVOC1,2048 worlds/batch,4M replay. Exactly one GPU worker, peak21653MiB. Step0 policy/state preserved and plan/source/config hashes verified. Raw checks and compact startup verification are in outputs/tier2_two_episode_monitor/startup_verification_20261027T201919Z.json. Do not interpret native assistance during skill acquisition as evaluation assistance.

Offline clean-reference scorer committed027a6ff: experiments/score_clean_reference.py. Seven CPU integration tests pass, including input noise retained in clean scoring, failed worlds retained, clean targets excluded from frame conversion, provenance/transform rejection and official visibility. It exactly reproduces prior episode5 clean-reference metrics and adds clean-reference temporal thirds and every-world error tails. This local official-code score uses saved mesh samples, so do not claim it is a hosted leaderboard score.

For scoring downloaded Tier2 arrays, use the main checkout's .soma-env/Scripts/python.exe, --dataset-input C:/Users/11534/OneDrive/Desktop/v2dchallenge/correction/data/tier2, --dataset-clean C:/Users/11534/OneDrive/Desktop/v2dchallenge/correction/data/tier1, --kit C:/Users/11534/OneDrive/Desktop/v2dchallenge/.worktrees/tier2-rl/vendor/submission-kit/v2d_submission_kit, the correct --episode, --arrays pointing to rollout_arrays.npz beside evaluation.json, and --reference-npz from that episode's corrected portable job. Write a new ignored --output file. Keep all clean data local; it is not part of the remote training archive. The runtime learner remains the immutable54e2a7c deployment, unaffected by this local scoring commit.
