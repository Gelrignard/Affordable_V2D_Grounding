# Third-party dependencies

NVIDIA Video to Data:
https://github.com/nvidia-isaac/video_to_data

The CPU prototype imports a separately downloaded pinned checkout. Its source
is not vendored here. The [compatibility patch](windows_cpu_preview.patch)
contains modifications to NVIDIA source under Apache License 2.0; the upstream
license is retained in [NVIDIA-LICENSE.txt](licenses/NVIDIA-LICENSE.txt). That original file also
contains the CC-BY-4.0 terms for upstream documentation/skills. Preserve the
original copyright and license notices when redistributing patched source.

Robot assets, pretrained models, and datasets are downloaded separately and
retain their respective upstream terms. They are not relicensed or included
by publishing this project. This notice does not assign a new project-wide
license to the original code in this repository.

The optional `video_processing/` package invokes a separately installed NVIDIA
CARI4D/SAM2/CoCoNet checkout pinned to
`193382a184308f6800331f3db829f76dc4294cfe`. Its adapters use the upstream MHR,
FoundationPose, and SOMA APIs; the upstream source and model weights are not
vendored. Preserve NVIDIA's license notices in that separate checkout. SAM2,
SAM 3D Body, MoGe, FoundationPose, CARI4D, and SOMA assets retain their respective
licenses and access conditions. A download instruction does not grant model or
dataset redistribution rights.
