"""Propagate actual image/depth refinements from explicitly reviewed native poses."""
import numpy as np

from .algorithm import MAX_ADJACENT_TRANSLATION_M, select_path, unresolved_gate


def generate(backend, plan, track, indices, settings):
    poses = np.asarray(track["fp_poses_cam"])[:, 0]
    carried = np.asarray(track.get("adaptive_carried", np.zeros(len(poses), bool)))
    pools, events, seeds = {}, [], {}
    for frame in indices:
        candidates = backend.measure(frame, poses[frame:frame + 1], "baseline")
        if carried[frame]:
            candidates[0].eligible = False
            candidates[0].reasons.append("baseline_is_carried")
        pools[frame] = candidates
    anchors = {a["id"]: a for a in plan["anchors"]}
    for key, anchor in anchors.items():
        frame = anchor["frame"]
        if carried[frame]:
            raise ValueError(f"Reviewed anchor {key} is a held pose, not an observed reconstruction")
        seed = backend.measure(frame, poses[frame:frame + 1], "reviewed_baseline_anchor", key)[0]
        events.append({"kind": "anchor", "id": key, "frame": frame, "eligible": seed.eligible, "reasons": seed.reasons})
        if seed.eligible:
            seeds[key] = seed
            if frame in pools:
                pools[frame].append(seed)
    def extend(key, interval, direction):
        start, end = interval["start"], interval["end"]
        frame0 = anchors[key]["frame"]
        previous, supported, unsupported = seeds[key].pose, True, 0
        stop = end + 1 if direction == 1 else start - 1
        for frame in range(frame0 + direction, stop, direction):
            if abs(frame - frame0) > settings["max_track_distance"]:
                events.append({"kind": "branch_stop", "anchor": key, "frame": frame, "reason": "distance_budget"})
                break
            pose = backend.track(frame, previous)
            candidate = backend.measure(frame, pose[None], "forward_track" if direction == 1 else "backward_track", key)[0]
            step = float(np.linalg.norm(pose[:3, 3] - previous[:3, 3]))
            if supported and candidate.eligible and step > MAX_ADJACENT_TRANSLATION_M:
                candidate.eligible = False
                candidate.reasons.append("adjacent_supported_step_over_1m")
            if start <= frame <= end:
                pools[frame].append(candidate)
            previous, supported = pose, candidate.eligible
            unsupported = 0 if supported else unsupported + 1
            if unsupported > settings["max_unverified_steps"] or "adjacent_supported_step_over_1m" in candidate.reasons:
                events.append({"kind": "branch_stop", "anchor": key, "frame": frame, "reason": "unsupported_refinement"})
                break
    for interval in plan["intervals"]:
        for key in interval["anchor_ids"]:
            if key not in seeds:
                continue
            if anchors[key]["frame"] < interval["end"]:
                extend(key, interval, 1)
            if anchors[key]["frame"] > interval["start"]:
                extend(key, interval, -1)
    # Immediate source neighbors are measured before applying hard seam limits.
    supported = np.zeros(len(poses), bool)
    for frame in set(indices) | {f for i in plan["intervals"] for f in (i["start"] - 1, i["end"] + 1) if 0 <= f < len(poses)}:
        value = pools[frame][0] if frame in pools else backend.measure(frame, poses[frame:frame + 1], "baseline")[0]
        supported[frame] = value.eligible and not carried[frame]
    selected, unresolved = select_path(pools, indices, poses, settings, supported)
    visible = np.zeros(len(poses), dtype=int)
    for frame in indices:
        visible[frame] = pools[frame][0].metrics["observed_visible_pixels"]
    reasons, uncertainty = unresolved_gate(unresolved, visible, settings)
    if not seeds:
        reasons.append("no_reviewed_anchor_passed_image_and_depth_gates")
    return pools, selected, unresolved, events, reasons, uncertainty
