# Package provenance

The external inference source is NVIDIA `video_to_data` commit
`193382a184308f6800331f3db829f76dc4294cfe`. Its CARI4D/SAM2 source and model licenses
are governed by that upstream repository and each dependency. The package imports
those implementations; it does not vendor model code or weights.

The local adaptation draws on these previously audited project-side files:

| Packaged component | Original project-side implementation | Scope |
| --- | --- | --- |
| `processing/core/native.py` | `ops/native_vision.py` | Stop before stage 07; new explicit paths and separate candidate branch |
| `processing/core/fp_diagnostics.py` | `reconstruction/fp_diagnostics.py` | Undefined centroid `+inf` to `NaN` only after estimation; unexpected infinities fail |
| `processing/core/sam2.py` | `reconstruction/sam2_entrypoint.py` | CPU 8-connected components; native SAM2 thresholds retained |
| `processing/output/native.py` | `reconstruction/export_artifacts.py` | Complete camera-space MHR/object/contact export; refinement comparison removed |
| `processing/output/mhr_soma.py` | `ops/convert_soma.py` | Existing MHR decode/SOMA fitting recipe; explicit weights root |
| `processing/output/geometry.py` | `ops/canonical_geometry.py` | Rigid geometry and optional 15-landmark alignment math |
| `correcting/core/{algorithm,backend}.py`, `correcting/output/native.py` | `ops/targeted_repair.py` | Ordinary RGB/depth scoring, native tracking, discrete selection and metadata-preserving merge |

These project-side files originated in the separate video preparation/repair
workspace. Its SSH automation, credentials, machine addresses, queue scripts,
dataset caches and per-episode exception policies are not dependencies of this
package. New input/output adapters replace its machine-specific paths and episode
manifests. No Tier 1 decoding or fetching code is included.

The new common correction supports **reviewed baseline anchors with ordinary
depth-supported forward/backward tracking only**. It omits automatic independent
registration, archived proposal import, RGB-only depth exceptions, directed mesh
landmark trials and hand-carry inference. It must not be described as an exact
reproduction of all archived repairs (including 002, 009, 013, 019, 025, 028 and
the older 017). Those artifacts keep their original per-case evidence, residuals
and methods in the external archive.

Source pin and SHA256 checks guard the upstream API. Compatibility adjustment is
performed in memory by a child process, so the checkout stays unchanged. Review
images and candidates are hash-bound; an operator's visual declaration remains a
declaration, not machine evidence that every visible 3D ambiguity is resolved.

Validation in this change: CPU contract checks and command parsing. Real GPU
inference for the new unified package has not been rerun. Historical execution
evidence supports the reused code, but is not represented as a GPU test of the
new entry points.
