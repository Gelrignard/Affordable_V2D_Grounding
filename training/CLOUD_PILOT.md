# AutoDL allocation for the two-episode pilot

Recommendation as of 25 October 2026: start with **one RTX 4090 24 GB**, at the
user's quoted approximately **2 CNY/hour**. The user also quotes 3090 at
**1.5 CNY/hour per card**. These are user-provided quotes, not a verified live
AutoDL tariff. L40, 4090 D and 3080 Ti quotes are not yet supplied.

| Option | Memory available to one job | How this implementation uses it | Allocation judgment |
| --- | --- | --- | --- |
| RTX 4090 | 24 GB | One simulator + frozen SONIC + residual learner | Preferred first profile at the quoted price |
| RTX 4090 D | 24 GB, standard NVIDIA specification | Same pipeline; same nominal memory capacity, fewer CUDA cores than 4090 | At equal price choose 4090; a discounted D is a reasonable alternative |
| L40 (not L40S) | 48 GB with ECC | More room for simulation worlds/replay if needed | Useful memory headroom, but not an established speedup for this code; consider if near the 4090 price or the profile is memory limited |
| 2 × RTX 3090 | 24 GB per job | Two independent jobs at once | Potential wall-time benefit; at 3 CNY/hour total it must exceed 1.5× single-4090 aggregate throughput to cost less for equal work |
| 2 × RTX 3080 Ti | 12 GB per job | Two independent jobs, potentially smaller simulation batches | More memory constrained; only attractive at a substantial discount and after a fit/throughput check |

Sources: [NVIDIA RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/),
[NVIDIA 4090 D specifications](https://www.nvidia.cn/geforce/graphics-cards/40-series/),
[NVIDIA L40 datasheet](https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf),
[NVIDIA 30-series comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/).
The memory specifications are verified; the allocation judgments are engineering
inferences, not measured simulator benchmarks. Do not substitute L40S throughput
or gaming/FLOP ratios for this workload's throughput.

Two cards do not pool their VRAM in this trainer. No distributed actor training,
gradient synchronization or NVLink is required. The plan's shard arguments assign
four independent jobs to each process. Do not run both processes on the same
CUDA device. With two separate single-card instances, each uses local `cuda:0`.

## Price comparison for the same work

Let one job take `t` hours on a 4090 and `k*t` hours on one 3090, for the same
transition/learning budget and settings. Eight similar-duration jobs give:

- One 4090: `8*t` wall hours, `16*t` CNY.
- Two 3090s: about `4*k*t` wall hours, `12*k*t` CNY.

The pair is faster if `k < 2`, and cheaper if `k < 4/3`. Equivalently, each 3090
must deliver more than 75% of 4090 throughput to be cheaper at these quotes.
Actual unequal job lengths, host contention and startup overhead affect the
parallel makespan. This criterion is more useful than assuming “two GPUs win.”
For any alternative, compare **total hourly price / measured aggregate
throughput**, then confirm equivalent quality. Lower cost per transition alone
does not prove lower cost per successful policy.

## First allocation and profile

Use the reduced recipe, not NVIDIA's full 2,048-world/10M-replay recipe. Ours
starts with 256 worlds, 131,072 CPU replay transitions and a small MLP world
model. A 24 GB GPU is a reasonable target, but has not yet been fit-tested.
For a 12 GB card, 64–128 worlds are profiling candidates, not validated settings.
Hold the chosen learner settings constant across the eight comparison runs.

Before renting, choose a host meeting the pinned runtime's **Linux / NVIDIA
driver >=580** requirement. A selectable CUDA image does not change the host
driver. The upstream runtime supports installation without Docker, useful for
AutoDL container instances. Start with Python 3.11 and the pinned dependencies;
do not assume an arbitrary preinstalled PyTorch image matches them.
See the [pinned FlashCHORD setup](https://github.com/nvidia-isaac/video_to_data/blob/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/robotic_grounding/flash_chord/README.md).

As provisioning headroom, prefer at least 16 vCPUs and 64 GB system RAM for one
GPU; two simultaneous jobs benefit from roughly twice the host resources.
These are practical targets, not measured minimums. This training path uses
CPU replay, so CPU/PCIe bandwidth and RAM also matter. Keep weights, compiler
caches, jobs and checkpoints on persistent storage. AutoDL's
[GPU selection guide](https://www.autodl.com/docs/gpu/) also distinguishes GPU,
CPU and host-memory selection.

1. Install and fetch actual SONIC weights; the local LFS pointers are not weights.
2. Run `python -m training doctor --runtime vendor/flash-runtime`.
3. Start a bounded profile of episode 16, then check episode 5 before the batch.
   Exclude first-time compilation from steady-state throughput, but include it
   in rental cost. Record total device peak usage (including simulator/ONNX),
   system RAM, transitions/s and learner time after warmup.
4. Check one real full-episode policy evaluation and its exporter. Then allocate
   the remaining eight-run budget using measured speed and observed learning.

The current plan caps each run at **2 training hours or 10M transitions**,
whichever comes first. Eight runs therefore cap the training loop at **16 GPU
hours**, about **32 CNY on the quoted 4090**, excluding setup/evaluation/storage
and small checkpoint/iteration overrun. This is a spending/stop scenario,
not a promise of successful policies within two hours.

One seed is an affordable paired pilot. The eight runs compare original versus
corrected references within Tier 1 and Tier 2; they do not independently measure
our learner against CHORD. The goal remains higher quality or lower cost, and
the viewer and current tests do not establish that result yet.
