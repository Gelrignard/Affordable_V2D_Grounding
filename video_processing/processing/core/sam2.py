"""Run SAM2 with the audited CPU fallback for component labeling.

Preserve 8-connectivity and per-component pixel counts; the native
fill_holes_in_mask_scores function still applies its original threshold.
"""
from __future__ import annotations

import runpy

import numpy as np


def components_numpy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import label
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("SAM2 component input must have shape N,1,H,W")
    labels = np.zeros(mask.shape, dtype=np.int32)
    counts = np.zeros_like(labels)
    for i in range(mask.shape[0]):
        lab, _ = label(mask[i, 0].astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
        areas = np.bincount(lab.ravel())
        areas[0] = 0
        labels[i, 0] = lab
        counts[i, 0] = areas[lab]
    return labels, counts


def components_cpu(mask):
    import torch
    labels, counts = components_numpy(mask.detach().to(device="cpu", dtype=torch.uint8).numpy())
    return torch.from_numpy(labels).to(mask.device), torch.from_numpy(counts).to(mask.device)


def install_cpu_components() -> None:
    import sam2.utils.misc
    sam2.utils.misc.get_connected_components = components_cpu
    print("SAM2 compatibility: CPU 8-connected component labels/counts; original hole-filling threshold retained", flush=True)


if __name__ == "__main__":
    install_cpu_components()
    runpy.run_module("v2d.sam2.lib.video_to_masks", run_name="__main__")
