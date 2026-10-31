"""Image/depth evidence and discrete selection adapted from the audited repair adapter."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

MAX_ADJACENT_TRANSLATION_M = 1.0

def depth_evidence(rendered_depth, observed_depth, rendered_mask, human_mask, object_mask):
    supported = rendered_mask & object_mask & ~human_mask & np.isfinite(observed_depth) & (observed_depth >= .001) & (rendered_depth >= .001)
    count = int(supported.sum())
    if not count:
        return {'depth_pixels': 0, 'depth_median_error_m': None, 'observed_depth_median_m': None}
    return {'depth_pixels': count,
            'depth_median_error_m': float(np.median(np.abs(rendered_depth[supported] - observed_depth[supported]))),
            'observed_depth_median_m': float(np.median(observed_depth[supported]))}

def assess(metrics, settings, landmark_normalized_errors=()):
    reasons = []
    for key, threshold in (('iou', 'min_iou'), ('precision', 'min_precision'), ('recall', 'min_recall')):
        value = metrics.get(key)
        if value is None or not np.isfinite(value) or value < settings[threshold]:
            reasons.append('low_' + key)
    if metrics.get('observed_visible_pixels', 0) < settings['min_visible_pixels']:
        reasons.append('insufficient_visible_support')
    if metrics.get('depth_pixels', 0) < settings['min_depth_pixels']:
        reasons.append('insufficient_depth_support')
    error, z = metrics.get('depth_median_error_m'), metrics.get('observed_depth_median_m')
    if error is None or z is None or not np.isfinite([error, z]).all():
        reasons.append('missing_depth_fit')
        depth_score = 0.
    else:
        limit = max(settings['depth_absolute_tolerance_m'], settings['depth_relative_tolerance'] * z)
        if error > limit:
            reasons.append('depth_disagreement')
        depth_score = max(0., 1. - error / limit)
    if any(not np.isfinite(x) or x > 1 for x in landmark_normalized_errors):
        reasons.append('reviewed_landmark_disagreement')
    score = .50 * float(metrics.get('iou', 0) or 0) + .20 * float(metrics.get('recall', 0) or 0) + .15 * float(metrics.get('precision', 0) or 0) + .15 * depth_score
    return not reasons, float(score) if np.isfinite(score) else -1., reasons

@dataclass
class Candidate:
    pose: np.ndarray
    origin: str
    metrics: dict
    eligible: bool
    score: float
    reasons: list = field(default_factory=list)
    anchor_id: str | None = None
    candidate_rank: int = -1
    registered: bool = False

def transition_cost(a, b, settings):
    distance = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = a[:3, :3] @ b[:3, :3].T
    angle = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    # A bounded ranking cost, never rejection against an old/stale pose.
    return settings['translation_transition_weight'] * min(distance / .1, 3.) + settings['rotation_transition_weight'] * min(angle / .5, 3.)

def select_path(pools, frames, baseline_poses, settings, baseline_image_supported=None):
    """Discrete candidate selection; no pose averaging, interpolation or smoothing."""
    selected, unresolved = {}, []
    runs, run = [], []
    for frame in frames:
        if any(c.eligible for c in pools.get(frame, [])):
            if run and frame != run[-1] + 1:
                runs.append(run); run = []
            run.append(frame)
        else:
            if run:
                runs.append(run); run = []
            unresolved.append(frame)
    if run:
        runs.append(run)
    def solve(run):
        if not run:
            return
        choices = [[c for c in pools[f] if c.eligible] for f in run]
        first = run[0]
        costs = np.array([-c.score + (transition_cost(baseline_poses[first - 1], c.pose, settings) if first else 0.) for c in choices[0]])
        if first and baseline_image_supported is not None and baseline_image_supported[first-1]:
            costs[[np.linalg.norm(c.pose[:3,3]-baseline_poses[first-1,:3,3]) > MAX_ADJACENT_TRANSLATION_M for c in choices[0]]] = np.inf
            if not np.isfinite(costs).any():
                unresolved.append(first); solve(run[1:]); return
        back = []
        for position, (left, right) in enumerate(zip(choices[:-1], choices[1:]), 1):
            matrix = np.array([[costs[i] + transition_cost(a.pose, b.pose, settings) - b.score
                                if np.linalg.norm(a.pose[:3,3]-b.pose[:3,3]) <= MAX_ADJACENT_TRANSLATION_M else np.inf
                                for i, a in enumerate(left)] for b in right])
            if not np.isfinite(matrix).any():
                # Keep the failed seam explicit; do not compare against an older
                # last-good pose or invent a bridge across the unsupported frame.
                solve(run[:position]); unresolved.append(run[position]); solve(run[position+1:]); return
            back.append(np.argmin(matrix, axis=1))
            costs = np.min(matrix, axis=1)
        if run[-1] + 1 < len(baseline_poses):
            costs += [transition_cost(c.pose, baseline_poses[run[-1] + 1], settings) for c in choices[-1]]
            if baseline_image_supported is not None and baseline_image_supported[run[-1]+1]:
                costs[[np.linalg.norm(c.pose[:3,3]-baseline_poses[run[-1]+1,:3,3]) > MAX_ADJACENT_TRANSLATION_M for c in choices[-1]]] = np.inf
                if not np.isfinite(costs).any():
                    unresolved.append(run[-1]); solve(run[:-1]); return
        index = int(np.argmin(costs))
        for position in range(len(run) - 1, -1, -1):
            selected[run[position]] = choices[position][index]
            if position:
                index = int(back[position - 1][index])
    for run in runs:
        solve(run)
    return selected, sorted(set(unresolved))

def serial_candidate(candidate):
    return {'origin': candidate.origin, 'anchor_id': candidate.anchor_id, 'candidate_rank': candidate.candidate_rank,
            'registered': candidate.registered, 'eligible': candidate.eligible, 'score': candidate.score,
            'rejection_reasons': candidate.reasons, 'pose_camera': candidate.pose.tolist(), 'metrics': candidate.metrics}

def unresolved_gate(unresolved, visible_pixels, settings, landmark_frames=()):
    visible = [f for f in unresolved if visible_pixels[f] >= settings['min_visible_pixels']]
    longest, run, previous = 0, 0, -2
    for frame in sorted(visible):
        run = run + 1 if frame == previous + 1 else 1
        longest = max(longest, run); previous = frame
    denominator = max(1, int(np.count_nonzero(np.asarray(visible_pixels) >= settings['min_visible_pixels'])))
    reasons = []
    if longest >= 15:
        reasons.append('persistent_visible_targeted_uncertainty_at_least_15_frames')
    if len(visible) / denominator > .10:
        reasons.append('visible_targeted_uncertainty_over_10pct_frames')
    if set(unresolved) & set(landmark_frames):
        reasons.append('reviewed_orientation_constraint_unresolved')
    return reasons, {'visible_unresolved_frames': visible, 'occluded_or_low_support_unresolved_frames': sorted(set(unresolved)-set(visible)), 'longest_visible_unresolved_run': longest, 'visible_unresolved_fraction': len(visible)/denominator,
                     'meaning':'Isolated or low-support uncertainty remains explicit and is not alone a rejection. Persistent visible disagreement and reviewed landmark contradictions remain blocking.'}

