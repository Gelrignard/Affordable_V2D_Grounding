# Optional NVIDIA dependency and compatibility patch

The standalone dynamics/contact optimizer does not require NVIDIA source.
The current G1 retargeter, book robot-trajectory repair, and robot viewers use
[NVIDIA Video to Data](https://github.com/nvidia-isaac/video_to_data) as an
optional external dependency. Download it locally into the ignored `upstream/`
directory when using those components. It is not included in this repository
or registered as a Git submodule.

The CPU prototype was tested against commit
`193382a184308f6800331f3db829f76dc4294cfe`. The published submission kit inspected
on 17 October identified a different evaluation version,
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`; prepare a separate evaluation runtime
as described in the [policy runtime handoff](training/POLICY_HANDOFF.md).

## What the patch changes

The [compatibility patch](windows_cpu_preview.patch) is a text diff against the
tested CPU source version. It changes four upstream files:

| File | Change |
|---|---|
| `motion_schema/reader.py` | Allow NumPy outputs on CPU without requiring Torch tensors |
| `motion_schema/schema.py` | Defer the Torch import until it is needed |
| `retarget/whole_body_kinematics.py` | Accept NumPy input without a runtime Torch dependency |
| `retarget/viser_playback.py` | Bind the historical viewer to the local machine |

These edits preserve the baseline IK objective. The source retains its original
license; see [third-party notices](THIRD_PARTY_NOTICES.md).

Patch SHA-256:
`ba1c354a36c7c3051f275b81cd9c64dca1015b2a6ceab504140c24d05a37705c`.

## Prepare a fresh optional checkout

Run from this repository's root. Use a fresh `upstream/` directory; do not
reinitialize or reset an existing working checkout.

Prevent automatic downloads of every Git LFS asset in the current shell:

```powershell
# Windows PowerShell
$env:GIT_LFS_SKIP_SMUDGE = "1"
```

```bash
# Linux/macOS shell
export GIT_LFS_SKIP_SMUDGE=1
```

Then use the same Git commands on either platform:

```text
git init upstream
git -C upstream remote add origin https://github.com/nvidia-isaac/video_to_data.git
git -C upstream config core.longpaths true
git -C upstream config core.autocrlf false
git -C upstream fetch --depth=1 --filter=blob:none origin 193382a184308f6800331f3db829f76dc4294cfe
git -C upstream sparse-checkout init --cone
git -C upstream sparse-checkout set robotic_grounding/source/robotic_grounding/robotic_grounding
git -C upstream checkout --detach 193382a184308f6800331f3db829f76dc4294cfe
git -C upstream apply --check --ignore-space-change ../docs/windows_cpu_preview.patch
git -C upstream apply --ignore-space-change ../docs/windows_cpu_preview.patch
python -m data_tools.fetch_preview_assets
```

The last command downloads only the selected robot/example assets, checking
their Git LFS hashes. Our Python adapters expect this local `upstream/` location.
There is no external dependency lock file or automatic source installation.

To check an already patched checkout without changing it:

```text
git -C upstream rev-parse HEAD
git -C upstream apply --reverse --check --ignore-space-change ../docs/windows_cpu_preview.patch
```

Confirm that the printed commit matches the CPU version above. Do not apply
the patch again when the reverse check succeeds.
