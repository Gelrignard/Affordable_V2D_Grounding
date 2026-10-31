"""Preserve native metadata and untouched frames while exporting a complete candidate."""
from copy import deepcopy
import numpy as np
VERSION = "reviewed-baseline-anchor-rgb-depth-v1"

METRICS_TO_NATIVE = {'iou': 'fp_full_image_iou', 'precision': 'fp_tolerant_precision',
                     'recall': 'fp_tolerant_recall', 'centroid_distance': 'fp_centroid_distance',
                     'rendered_visible_pixels': 'fp_rendered_visible_pixels',
                     'rendered_visibility': 'fp_rendered_visibility'}


def write_measurements(track, frame, metrics):
    if int(metrics['observed_visible_pixels']) != int(track['fp_observed_visible_pixels'][frame]):
        raise ValueError('Observed segmentation support changed')
    for key, destination in METRICS_TO_NATIVE.items():
        value = metrics[key]
        if key == 'centroid_distance' and np.isinf(value):
            # The native silhouette metric uses +inf when either centroid is
            # undefined. Native stored diagnostics require NaN for that case.
            # Keep raw candidate metrics/eligibility intact for the evidence.
            if value > 0 and (int(metrics['observed_visible_pixels']) == 0
                              or int(metrics['rendered_visible_pixels']) == 0):
                value = np.nan
            else:
                raise ValueError('Unexpected infinite centroid with defined silhouette support')
        track[destination][frame] = value
    track['fp_selected_iou'][frame] = metrics['iou']
    track['fp_selected_visibility'][frame] = metrics['rendered_visibility']

def merge_selected(original, selected, indices, unresolved, c2w, native):
    merged = deepcopy(original); count = len(original['frames'])
    carried = np.asarray(original.get('adaptive_carried', np.zeros(count, bool))).copy()
    reset = np.asarray(original.get('adaptive_reinitialized', np.zeros(count, bool))).copy()
    reasons = list(merged['metadata']['object_pose_fallback_reasons'])
    rgb_history = np.asarray(original.get('targeted_rgb_only', np.zeros(count, bool))).copy()
    depth_uncertainty = np.asarray(original.get('targeted_depth_uncertain', np.zeros(count, bool))).copy()
    for frame, candidate in sorted(selected.items()):
        if candidate.origin == 'baseline':
            continue
        pose = candidate.pose; world = c2w @ pose
        merged['fp_poses_cam'][frame, 0] = pose
        merged['obj_pose_world'][frame] = world
        merged['obj_rot_init'][frame] = world[:3, :3]
        merged['obj_t_init'][frame] = world[:3, 3]
        metrics = candidate.metrics
        write_measurements(merged, frame, metrics)
        depth_uncertain = bool(metrics.get('depth_uncertainty_reasons'))
        rgb_history[frame] = 'rgb_only' in candidate.origin
        depth_uncertainty[frame] = depth_uncertain
        merged['fp_reliable'][frame] = not depth_uncertain
        merged['fp_candidate_count'][frame] = 1
        merged['fp_selected_candidate_index'][frame] = 0
        merged['fp_retry_count'][frame] = int(candidate.registered)
        merged['fp_translation_threshold_m'][frame] = np.nan
        merged['fp_selected_translation_distance_m'][frame] = np.linalg.norm(pose[:3, 3] - merged['fp_poses_cam'][max(0, frame-1), 0, :3, 3])
        source = native['PoseSource'].REACQUISITION_CANDIDATE if candidate.registered else native['PoseSource'].FOUNDATIONPOSE_TRACK
        for key in ('fp_confirmed_pose_source', 'fp_emitted_pose_source'):
            merged[key][frame] = int(source)
        merged['fp_pending_pose_source'][frame] = int(native['PoseSource'].NONE)
        merged['fp_tracking_state'][frame] = int(native['TrackingState'].TRACKING)
        for key, value in (('fp_gap_length', 1), ('fp_confirmation_count', 1), ('fp_rejection_mask', 0), ('fp_reacquisition_latency', 1)):
            merged[key][frame] = value
        carried[frame] = False; reset[frame] = candidate.registered
        reasons[frame] = 'targeted_rgb_only_depth_unverified' if depth_uncertain else ''
    merged['adaptive_carried'] = carried
    merged['adaptive_reinitialized'] = reset
    merged['targeted_repair_selected'] = np.array([f in selected and selected[f].origin != 'baseline' for f in range(count)], bool)
    merged['targeted_repair_unresolved'] = np.array([f in unresolved for f in range(count)], bool)
    merged['targeted_rgb_only'] = rgb_history
    merged['targeted_depth_uncertain'] = depth_uncertainty
    merged['fp_reliable'][unresolved] = False
    metadata = merged['metadata']
    metadata['original_track_metadata'] = deepcopy(original['metadata'])
    metadata['object_pose_fallback_reasons'] = reasons
    metadata['object_pose_fallback_count'] = sum(bool(x) for x in reasons)
    metadata['reliable_frame_count'] = int(merged['fp_reliable'].sum())
    metadata['retry_frame_count'] = int(np.count_nonzero(merged['fp_retry_count']))
    metadata['retry_attempt_count'] = int(merged['fp_retry_count'].sum())
    metadata['foundationpose_sequence_mode'] = VERSION
    metadata['selection_pipeline'] = VERSION
    for key in ('registration_call_count', 'tracking_call_count', 'retry_exhausted_count'):
        metadata.pop(key, None)
    outside = np.ones(count, bool); outside[indices] = False
    for key, value in original.items():
        if (isinstance(value, np.ndarray) and value.ndim and len(value) == count
            and key not in ('targeted_repair_selected', 'targeted_repair_unresolved')):
            if not np.array_equal(value[outside], merged[key][outside], equal_nan=True):
                raise AssertionError(f'Preserved frame array changed outside target intervals: {key}')
    return merged

