"""Repair an upstream missing-value representation after pose estimation."""
import numpy as np

REPAIR_ID = "foundationpose-centroid-missing-value-v1"


def normalize_missing_centroids(result):
    values = np.asarray(result["fp_centroid_distance"])
    missing = np.isposinf(values)
    observed = np.asarray(result["fp_observed_visible_pixels"])
    rendered = np.asarray(result["fp_rendered_visible_pixels"])
    # measure_visible_silhouettes initializes +inf and only computes a distance
    # when both visible centroids and a positive observed bbox diagonal exist.
    expected_missing = (observed <= 1) | (rendered == 0)
    if np.isneginf(values).any() or np.any(missing & ~expected_missing):
        raise ValueError("Unexpected infinite centroid distance; refusing to hide a numerical failure")
    fixed = values.copy()
    fixed[missing] = np.nan
    result["fp_centroid_distance"] = fixed
    result["metadata"]["diagnostic_serialization_repair"] = {
        "id": REPAIR_ID,
        "missing_centroid_frame_indices": np.flatnonzero(missing).tolist(),
        "meaning": "No defined visible centroid distance; +inf serialized as NaN. Pose estimation, ranking, rejection and reliability flags are unchanged.",
    }
    print(f"FOUNDATIONPOSE_DIAGNOSTIC_REPAIR missing_centroid_frames={int(missing.sum())}", flush=True)
