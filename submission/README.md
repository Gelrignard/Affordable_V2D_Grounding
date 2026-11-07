# Submitted-container launcher

This directory publishes the corrected `/submission/sunfish_container.py` from
`sunfish-v2d-submission:20261103` and an offline Tier 1 episode 0 replay script.
The existing image needs this launcher mounted over its original file, or included
in a separately validated replacement image. Updating this repository alone does
not change an already downloaded image.

## What changed

The original launcher resolved the Python executable's symlinks, turning
`/opt/v2d-policy-env/bin/python` into `/usr/bin/python3.12`. That selected the system
Python for both evaluation and NVIDIA library discovery. Keeping the venv launch
path fixes dependency imports and lets the existing environment setup expose the
installed CUDA libraries to ONNX Runtime.

The corrected launcher:

- Preserves the interpreter's symlink path while retaining normal asset path
  resolution.
- Honors an explicit `EVAL_PYTHON`, using that interpreter for evaluation, export,
  and NVIDIA library discovery. It accepts an absolute path, a path relative to
  the submission root, or a command name found on the container's `PATH`.
- Finds NVIDIA libraries through the selected interpreter's package paths and
  `purelib`/`platlib`, using the evaluation subprocess's `PYTHONPATH`. Existing
  `LD_LIBRARY_PATH` entries are retained. An absent pip NVIDIA directory is not
  itself an error; system or Conda libraries can provide the dependencies.
- Does not reject environments by GPU model, driver version, or venv directory
  name. The helper script warns about a different image ID and continues with the
  selected image.

Missing dependencies, unavailable CUDA support, invalid inputs, and provenance
mismatches still fail. The launcher does not install dependencies or silently
replace GPU evaluation with CPU evaluation. Policy weights, evaluator logic,
fixed-world export, and provenance checks are unchanged.

## Replay the existing image

Use a host with Docker, NVIDIA Container Toolkit, and a working GPU. Load the
original submission archive if the image is not already present:

```bash
zstd -dc Sunfish-v2d-submission.tar.zst | docker load
```

The diagnosed archive SHA-256 is
`59de3076311bfbd113e5f30ef33318f22f1409cbc3124fa2a0f5bcddec8a23c3`.
The corresponding tested image ID is
`sha256:2f78529c9100d710c1fe58d1f11dfb4795839cc066805de630c33f2f9f883b15`.
These identify the diagnosed artifact; they are not driver compatibility checks.

From the repository root, choose an output directory that does not yet exist:

```bash
bash submission/rerun-tier1-episode0.sh \
  /absolute/path/to/dataset /absolute/path/to/new-output
```

An optional third argument selects another image tag or ID. The script uses
`--pull=never --gpus all --network none --read-only`, mounts the dataset and this
launcher's source read-only, and keeps the image's entrypoint. Temporary caches
and the new output directory are writable. It records image metadata, the exact
command, launcher hash, stdout, stderr, and exit code. Evaluator logs and exported
artifacts are under `NEW_OUTPUT_DIRECTORY/output/tier1/`.

If a different compatible interpreter already exists inside the image, select it
explicitly:

```bash
EVAL_PYTHON=/opt/another-env/bin/python \
  bash submission/rerun-tier1-episode0.sh /dataset /new-output compatible-image:tag
```

The original `run.sh` also supports `KIT_PYTHON` for the launcher and official
packer/verifier. Normally its image default is sufficient. Both interpreters must
have the dependencies their respective commands need; no installation occurs.

## Include the fix in a replacement image

The final submission image must contain this file at
`/submission/sunfish_container.py`. In the final image's build definition, with
this repository as the build context, the relevant copy is:

```dockerfile
COPY submission/sunfish_container.py /submission/sunfish_container.py
```

Keep the original `/submission/run.sh` entrypoint and the packaged runtime,
inputs, weights, manifests, and frozen evaluator checkouts. The repository's
`training/Dockerfile` is not the final submitted image's build definition.
No replacement image is published by this source update.

The packaging-fix revision is separate from the recorded producing revisions:

- Tier 1: `8cc63a6b6da299aad9d0e671f4e29eb1f8179afc`
- Tier 3: `bf5b4f15b78fec8a222b607a83a1b0c69138037c`

Do not replace the manifest/CSV producing revisions or the frozen checkouts with
this packaging revision. Their hashes remain part of the existing verification
contract. After rebuilding, test the new image's original entrypoint without the
launcher bind mount before publishing it.

## Validation scope

The exact launcher and helper script published here were checked on 2026-11-07:

- 12 interpreter/library-layout checks and five helper-script tests passed. The
  helper-script tests used a fake Docker CLI to verify command construction and
  exit-code handling.
- All 90 episode plans and 402 historical path mappings across the three tiers
  passed preflight checks.
- The actual SONIC encoder passed CUDA session creation and GPU I/O binding on
  `cuda:0` (RTX 3090 Ti) and `cuda:1` (RTX 2080 Ti), including a renamed venv path.
  Actual providers were checked before binding; the Warp stream and CUDA graph
  settings were retained, with JAX initialized first.
- The original `run.sh` entrypoint completed Tier 1 episode 0 offline: 16 worlds,
  778 steps, fixed world 0 export, provenance checks, and exit code 0 in about
  270 seconds. All 90 original actor weights and the checked evaluator/runtime
  files retained their hashes.

GPU validation used the original image filesystem with bubblewrap network
isolation and host driver 595.99.02. Native Docker and driver 580 were not tested.
The renamed environment used the same installed dependencies, not arbitrary
package versions. Full GPU replay of all episodes and final pack/verification
remain separate validation steps requiring the complete challenge dataset.
