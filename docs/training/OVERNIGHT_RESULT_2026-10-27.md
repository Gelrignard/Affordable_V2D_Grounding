# Overnight policy refinement — 27 October 2026

The refinement worked on the tested box-rotation episode. Keep **native FlashSAC with the object-relative hand objective, recovery termination, and the coarse-plus-fine object-position reward**. The selected checkpoint is **500000768**. Training stopped early after later weights regressed; the instance remains powered on.

## Results

All results below use the same 16 worlds and 1,003 frames, starting at frame zero, with zero global/reset assistance, no settling, no within-rollout reset, and no excluded worlds. CPU recomputation with NVIDIA's released scoring code and clean reference reproduces the selected checkpoint's scores. These are local results for episode 5, not a leaderboard submission; AUC uses local mesh samples.

| Checkpoint / decision | ADD-AUC ↑ | MPPE cm ↓ | SP-SR ↑ | MP-SR ↑ |
|---|---:|---:|---:|---:|
| Existing CHORD SAC parent, 349999104 | .422089 | 14.557 | 0% | 0% |
| First refinement, 409997312 | .573927 | 5.976 | 81.25% | 12.5% |
| Precision screen, 469995520 — fallback | .741514 | 2.206 | 100% | 93.75% |
| **Selected precision midpoint, 500000768** | **.770692** | **2.102** | **100%** | **93.75%** |
| Later precision endpoint, 529993728 — rejected | .749821 | 4.610 | 87.5% | 75% |

The selected policy reduces MPPE by about 86% relative to the paired parent. It stays below 3 cm object-origin error for 84.04% of world-frames. Two brief excursions exceed 10 cm, then recover; the largest is 11.04 cm. No rollout crosses the pelvis or object-orientation failure thresholds. Native CHORD wrist-imitation success remains zero because its wrist gates differ from the object metrics.

Selection does involve a small tradeoff: compared with the precision screen, final 200-frame MPPE changes 3.199→3.386cm. Across the full sequence, AUC and MPPE improve, success rates are preserved, and >10cm world-frames fall 46→14. The precision screen is retained separately as fallback.

![All-world object tracking comparison](C:/Users/11534/.codex/worktrees/policy-overnight/v2dchallenge/outputs/overnight_analysis/parent_vs_selected_precision.png)

## What changed and what the experiments support

The first refinement softened exact hand imitation into contact-phase object-relative guidance, allowed brief recovery before training termination, and emphasized object pose while retaining body stabilization. This improved the parent substantially. It preserved observations, actions, SONIC control, physics and object-target timing.

The strongest focused improvement was adding precision to the broad object-position reward. With position error `e` in metres, the raw term becomes:

`(2/3) * exp(-(e / .20)^2) + (1/3) * exp(-(e / .05)^2)`

The outer position weight remains 3, so its peak is unchanged. The broad component still supports recovery; the 5 cm component gives centimetre-scale accuracy more influence. From the same 409997312 parent and 60M-transition budget, unchanged broad-reward training scored AUC .566644/MPPE11.057cm, while the precision mixture scored .741514/2.206cm. That result supports this change on the tested episode.

An all-frame-zero consolidation was rejected: AUC fell .573927→.523527 and MP-SR fell 12.5%→0. Continued broad-reward refinement also plateaued and developed failure tails. Neither branch was resumed. The later precision continuation improved typical precision but introduced object loss and body collapse in a few worlds. Optimizer records remained finite with rare isolated mixed-precision skips. We evaluated one already-saved midpoint for 50 seconds, selected it, and stopped training rather than extending the degraded endpoint.

## Compute and timing

- Additional GPU wall time charged to the canonical ledger: **8,554.353seconds = 2h22m34s**, including evaluations, failed startup attempts and the memory diagnostic.
- Candidate1 training: 4,987.381seconds; candidate2 training: 3,116.563seconds. Two implementations were used; no third candidate or parameter sweep.
- The selected lineage adds roughly 66 minutes and 150M transitions to the existing 350M parent. The parent itself had about 4.25 hours of prior training. **This establishes useful warm-start refinement; a complete 2–3 hour run from scratch remains untested.**
- Stable configuration: 4M GPU replay, 2,048 worlds/batch, JAX memory fraction .60, roughly 44k transitions/second and 21,687 MiB peak GPU memory. Initial larger allocations hit the memory guard and are included in the ledger.
- Last GPU job finished at **06:07:51 New York time**. No overnight GPU job remains active. The instance was left powered on as requested.

## Retargeting and scoring

Different robot retargeting does **not** directly reduce Track2's numerical score: the metrics compare achieved object poses against the clean multiview object reference, not robot joints or hands against NVIDIA's retargeter. The fixed G1+Dex3 and learned whole-body-policy requirements are separate. The packer uses a fixed reference-to-input coordinate conversion; it does not fit achieved motion to erase tracking error. See the [official challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/), [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf), and [submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip).

For this episode, original and corrected object references are identical, and the evaluator reference is exactly the released clean trajectory under one rigid transform. The gains therefore do not come from an easier object target. This remains a Tier 1-trained episode result; it does not establish a Tier 2 or 30-episode aggregate. The inherited 30 Hz source / 50 Hz evaluation timing is recorded in the audit and was unchanged by refinement.

## Restore and inspect

- Tested source commit: **481b5233051da5093337d738759df460048b3dcf**. Branch: `codex/policy-overnight-precision-20261027`; later commits record decisions and this report.
- Selected remote pair: `/root/autodl-tmp/v2d/overnight_policy_20261027/candidate2_precision_extension1/checkpoints/{policy,state}_500000768.safetensors`.
- Local pair and verification: [backup manifest](C:/Users/11534/.codex/worktrees/policy-overnight/v2dchallenge/outputs/overnight_best/candidate2_midpoint/local_backup_manifest.json).
- Policy SHA256: `97892c1d0e888eb9dee8d72e31ef90a7812198f2ee01d4f1f80962fc7da16a30`.
- Compact-state SHA256: `f2d3600bd91d62b8df5ffb6343d51349f2054caf2acfab2ddb5f4301da152b07`.
- [Full rollout diagnosis](C:/Users/11534/.codex/worktrees/policy-overnight/v2dchallenge/outputs/overnight_analysis/candidate2_midpoint_cpu_diagnosis.json), [clean-reference recomputation](C:/Users/11534/.codex/worktrees/policy-overnight/v2dchallenge/outputs/overnight_analysis/candidate2_midpoint_clean_reference.json), and [durable experiment log](C:/Users/11534/.codex/worktrees/policy-overnight/v2dchallenge/docs/training/OVERNIGHT_2026-10-27.md).

Original, first-refinement and precision-screen backups remain intact. Remote best/history/decision files identify the selected midpoint. The compact state retains model, optimizer and normalizer state; replay is not saved and must be refilled with the restored policy on any later continuation. No changes were published or applied to another agent's checkout.
