# AutoDL allocation for the two-episode pilot

Recommendation as of 25 October 2026: start with **one RTX 4090 24 GB**, at the
user's quoted approximately **2 CNY/hour**. The user also quotes 3090 at
**1.5 CNY/hour per card**. These are user-provided quotes, not a verified live
AutoDL tariff. L40, 4090 D and 3080 Ti quotes are not yet supplied.

Allocation update, 25 October: the user purchased the instance for a full week
and wants it kept powered on. Hourly figures below are earlier allocation
comparisons, not additional charges for the prepaid week. Track elapsed GPU
time and progress toward the 90 episode/tier policies instead.

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
uses 256 worlds, 131,072 GPU replay transitions and a small MLP world
model. The episode-16 GPU replay profile fits in 4.4 GiB on the provisioned
4090; CPU replay profiles of both episodes fit in about 2.0 GiB of device memory.
For a 12 GB card, 64–128 worlds are profiling candidates, not validated settings.
Hold the chosen learner settings constant across the eight comparison runs.

The upstream CUDA 13 / JAX training image requires **Linux / NVIDIA driver
>=580**. Our PyTorch learner does not use JAX. A CUDA 12 installation can reuse
a compatible preinstalled PyTorch, but must verify actual Warp kernels and
SONIC GPU inference before running physics. In particular, the default
`onnxruntime-gpu==1.28.0` wheel downloaded on this host requires CUDA 13;
Microsoft also publishes a CUDA 12 build of the same version. Merely seeing
`CUDAExecutionProvider` in the available-provider list does not prove it loaded.
A selectable CUDA image does not change the host driver. The upstream runtime
supports installation without Docker and Python 3.11/3.12.
See the [pinned FlashCHORD setup](https://github.com/nvidia-isaac/video_to_data/blob/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/robotic_grounding/flash_chord/README.md).

As provisioning headroom, prefer at least 16 vCPUs and 64 GB system RAM for one
GPU; two simultaneous jobs benefit from roughly twice the host resources.
These are practical targets, not measured minimums. CPU replay remains an
option for other machines; its CPU/PCIe bandwidth and RAM also matter. Keep weights, compiler
caches, jobs and checkpoints on persistent storage. AutoDL's
[GPU selection guide](https://www.autodl.com/docs/gpu/) also distinguishes GPU,
CPU and host-memory selection.

1. Install and fetch actual SONIC weights; the local LFS pointers are not weights.
2. Run `python -m training doctor --runtime vendor/flash-runtime`.
   Then run `python -m training.gpu_preflight --runtime vendor/flash-runtime`.
   This compiles a Warp CUDA kernel, checks the two exact SONIC model hashes,
   and runs both networks while rejecting CPU fallback.
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

## Provisioned AutoDL instance

The instance inspected on 25 October 2026 has one RTX 4090 (24,564 MiB), driver
570.124.04, 16 CPU cores of cgroup quota, 120 GiB RAM, and a 50 GB data disk.
The base environment supplies Python 3.12.3 and PyTorch 2.7.0+cu128. We reuse
that PyTorch in a `--system-site-packages` virtual environment rather than
download another CUDA/PyTorch stack. This differs from the local CPU test
environment's PyTorch 2.8; keep the difference in experiment provenance.

Deployment layout:

- `/root/autodl-tmp/v2d/project`: training code, pinned runtime, eight prepared jobs.
- `/root/autodl-tmp/v2d/.venv`: project dependency overlay; base environment is preserved.
- `/root/autodl-tmp/v2d/cache`: pip downloads, hashed LFS assets and CoACD cache.
- `/root/autodl-tmp/v2d/logs`: installation, tests, GPU checks and resource logs.
- `/root/autodl-tmp/v2d/activate.sh`: environment variables and project directory.

The initial upload is 33,950,284 bytes, with identical immutable job files
deduplicated. The transfer manifest records every included file hash, source
commit `e892b04`, and runtime commit
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. Subsequent code fixes are small
overlays and must be recorded separately. Only the two SONIC networks and G1
assets are fetched from NVIDIA; no Tier 3 data or full CHORD training corpus
is downloaded. Every LFS object is checked against its Git SHA-256 and size.

Ordinary Python dependencies use `https://mirrors.aliyun.com/pypi/simple` and
the versions in the pinned runtime's `pyproject.toml`. ONNX Runtime is the
exception: install its CUDA 12 build from Microsoft's feed, with dependencies
already present:

```bash
source /root/autodl-tmp/v2d/activate.sh
python -m pip install --force-reinstall --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.28.0
python -m training.gpu_preflight --runtime vendor/flash-runtime
```

For this CPython 3.12 Linux CUDA 12 wheel, the feed's SHA-256 is
`6e33c999981b746684fea80f3cc3d42dd651a76431a7bb4b03368a58e2db5b88`.
Keep its cached wheel for restarts; do not repeat this download on a configured
instance. The Warp 1.14.0 wheel uses CUDA 12.9; a compiled CUDA kernel executed
on this host's 570 driver. Do not infer complete simulator compatibility from
that kernel alone: the physical smoke run and full evaluation are separate checks.

The training plan remains `outputs/policy_pilot_two/plan.json`: episodes 5 and
16, both tiers, both reference variants, one learner and seed 0. After physical
validation, a bounded independent run can be started with:

```bash
source /root/autodl-tmp/v2d/activate.sh
python -m training batch --plan outputs/policy_pilot_two/plan.json \
  --runtime vendor/flash-runtime --max-jobs 1 --evaluate --device cuda:0
```

The command runs the first job. Use `--start N` for a later planned job, and
`--max-jobs 8` only when intending to run the full eight-run budget. Existing
output directories are rejected rather than overwritten. Use `nohup` or a
terminal multiplexer for long runs; retain stdout, checkpoints and evaluations
on the data disk. AutoDL instance billing continues while it is running, even
when this command has finished.

## Measured validation results

All three profiles below used 256 worlds, 250,112 collected transitions,
921 SAC updates and 464 world-model updates. Throughput excludes the first
32,768 transitions. Total GPU peaks were sampled with `nvidia-smi`, including
SONIC and physics allocations, rather than PyTorch allocations alone.

| Setup profile | Replay | Training loop | Transitions/s after warmup | Total GPU peak |
| --- | --- | --- | --- | --- |
| Tier 1, episode 16, original | CPU | 48.7 s | 4,913 | 2,051 MiB |
| Tier 2, episode 5, corrected | CPU | 51.8 s | 4,620 | 2,075 MiB |
| Tier 1, episode 16, original | GPU | 29.3 s | 8,218 | 4,495 MiB |

GPU replay improved throughput about 1.67x on the tested episode. The two
episode-16 profiles enabled imagined updates in 17 and 16 collection rounds,
respectively; the episode-5 model did not pass its validation threshold within
this short profile. Replay placement also changes the RNG stream used for
sampling, so this is a practical throughput comparison, not identical stochastic
training or evidence of improved policy quality. The 4090 recipe and both
deployed/local eight-run plans now use GPU replay. The previous CPU plan is
preserved as `plan.before_gpu_replay.json`. Defaults for other machines still
allow CPU replay.

The first physical eight-world smoke run took 333.7 seconds including cold
startup/compilation, then collected 2,048 transitions in a 2.4-second loop.
Warm startup plus the episode-16 GPU replay profile took 58.6 seconds overall.
Caches are retained on the data disk.

Full evaluations completed for both short-trained checkpoints, with no resets
or object assistance: episode 16 recorded 794 physical control steps and all
477 source frames; episode 5 recorded 1,671 steps and all 1,003 source frames.
Both exported Parquet files were accepted by the official submission kit reader.
The trajectories stayed finite, but both policies triggered tracking termination
and continued recording afterward. Their mean object-position errors against
their respective tier inputs were 0.490 m and 0.928 m. These are setup
checkpoints, not successful submissions or a correction-quality comparison.

Sixteen tests pass on the cloud host, including all eight prepared jobs. The
physical test exposed an independent interpolation/FK mismatch, now fixed by
recomputing derived palm poses after control-rate resampling. Joint/root and
observed object trajectories remain unchanged. The adapter rejects SONIC CPU
fallback, and the GPU preflight confirms both networks execute on CUDA.

If measured throughput holds, eight fixed 10M-transition runs would take roughly
3–5 training hours (about 6–10 CNY at the quoted price), plus initialization and
evaluation. Only episode 16 has been profiled with GPU replay; later training
and different contact behavior can change throughput. This estimate concerns
the fixed transition budget, not the time required to learn successful policies.
The existing 2-hour/run wall cap remains. Environment validation finished before
the full eight-run comparison was started; its execution record is below.

Detailed reports and exported trajectories are retained on the cloud under
`project/outputs/cloud_*`; a small local copy is under
`outputs/cloud/validation_results`. Large checkpoints and control traces remain
on the cloud instance.

## Eight-run execution and notifications

The full queue started at **2026-10-25 18:34:33 UTC**, using learner commit
`484cff143df19cec4efdd7e5dd3fc55f232c2bc8`. It trains and then evaluates each of
the eight entries in `outputs/policy_pilot_two/plan.json`, with 10M transitions
or a 2-hour training cap per entry. Every entry initializes independently.
The command is the batch command above with `--max-jobs 8 --evaluate`.

The detached launcher is `/root/autodl-tmp/v2d/run_pilot.sh`; it holds a `flock`
lock and records its PID, start/end timestamps and exit code under `logs/`.
`logs/pilot.log` records the combined training/evaluation output. A failed
subprocess stops the queue instead of silently skipping that experiment.
The first run exceeded 2.8M transitions at approximately 7,000 transitions/s;
its learned dynamics passed the held-out validation gate and enabled imagined
actor updates. This verifies learning is active, not manipulation success.

**User instruction: keep the instance powered on and notify immediately on
completion or failure. No automatic shutdown is installed.** A ten-minute
heartbeat named `Watch AutoDL policy pilot` checks this chat's running queue.
It reports completion, failure, missing queue processes, stalled activity, or
a lost monitoring connection. Routine unchanged checks remain quiet. The
heartbeat is paused after the final completion/failure notice.

`training/cloud_status.py` provides a standard-library, read-only JSON snapshot.
On the server it lives at `/root/autodl-tmp/v2d/cloud_status.py`, outside the
immutable learner deployment. Its dedicated SSH key is restricted by
`authorized_keys` to that one command, without shell or forwarding access.
The private key stays outside the repository. On the configured Windows host:

```powershell
ssh -T -i "$env:USERPROFILE/.ssh/v2d_autodl_monitor" `
  -o BatchMode=yes -o ConnectTimeout=15 -p 23073 root@connect.bjb1.seetacloud.com
```

The snapshot includes all eight runs, latest learning metrics, evaluation
results, exit status and approximate queue cost at the user-quoted price.
Queue completion requires all eight evaluation reports; an exit code alone
does not establish completion. A finite trajectory does not establish successful
tracking. The endpoint retains the original hourly-equivalent cost field for
comparison; it does not represent an extra charge on the prepaid allocation.

At approximately 7,000 transitions/s, a 10M-transition budget takes about
24 minutes of learning per policy. Ninety such policies would take about
36 hours of learning sequentially; plan roughly 40–60 hours for a first pass
including initialization and evaluation. This extrapolates the first pilot
episode, not a measured result for all tiers or a time-to-success guarantee.
One week leaves room for diagnosis and targeted retries. The current pilot
continues sequentially. Before expanding it, benchmark two independent learners
sharing the 4090 against sequential aggregate throughput; fitting in VRAM alone
does not imply faster training. The 90-policy submission pass needs one selected
variant per episode/tier, not both pilot comparison variants.

## China-side transfers

Research downloads ran concurrently with training. Use AutoDL's documented
academic accelerator in a download-only subshell:

```bash
(source /etc/network_turbo; git clone --depth 1 --filter=blob:none --no-checkout URL)
```

Use sparse checkout and skip Git LFS assets when only inspecting source. The
checkpoint download used `hf-mirror.com`, a pinned revision, range resumption,
and final SHA-256/size checks. It transferred 221,362,447 bytes in 69.8 seconds
directly onto the China instance. Only an 8.6 KB evidence archive came back to
the local computer; the weights remain on the server. Python packages use the
Aliyun index already configured above, with cached wheels and simulator assets.
See [AutoDL academic acceleration](https://api.autodl.com/docs/network_turbo/).

For future large private inputs, AutoPanel can transfer between the instance
and an authorized cloud drive or OSS account without routing the transfer through
the browser's computer. No cloud-drive account has been connected here. Regional
AutoDL NAS also permits uploads while the GPU instance is off and sharing between
instances in the same region; copy hot training data to `autodl-tmp` for faster
reads. See [AutoPanel transfers](https://api.autodl.com/docs/netdisk/) and
[regional NAS](https://api.autodl.com/docs/nas/).

The reproduction survey, checkpoint contract, and reuse conclusions are in
[SUPERMIMIC_REUSE.md](SUPERMIMIC_REUSE.md). Raw evidence is retained under
`outputs/cloud/research_evidence/` locally and `research/` on the server.
