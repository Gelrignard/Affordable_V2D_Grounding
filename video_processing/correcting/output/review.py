"""Save baseline/candidate RGB overlays for real, separately recorded human review."""
import numpy as np


def representative_frames(frames):
    """First/middle/last of each unresolved run; full diagnostics remain separate."""
    runs, run = [], []
    for frame in sorted(set(frames)):
        if run and frame != run[-1] + 1:
            runs.append(run)
            run = []
        run.append(frame)
    if run:
        runs.append(run)
    return {f for run in runs for f in (run[0], run[len(run) // 2], run[-1])}


def render_review(backend, original, merged, frames, output):
    cv2 = backend.native["cv2"]
    output.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        rgb, _, _, _, K = backend.frame(frame)
        panels = []
        for label, track in (("baseline", original), ("candidate", merged)):
            pose = np.asarray(track["fp_poses_cam"])[frame, 0] @ backend.inverse_tf
            depth = backend.rendered_depth(pose[None], K, rgb.shape[:2])[0]
            panel = rgb.copy()
            mask = depth > .001
            panel[mask] = (.65 * panel[mask] + .35 * np.array([0, 255, 180])).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (255, 230, 0), 2)
            # Axes expose orientation changes even for similar silhouettes.
            size = max(.01, float(backend.estimator.diameter) * .35)
            points = np.array([[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
            camera = points @ pose[:3, :3].T + pose[:3, 3]
            if np.all(camera[:, 2] > .001):
                projected = camera @ K.T
                pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
                for i, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 100, 255)), 1):
                    cv2.line(panel, tuple(pixels[0]), tuple(pixels[i]), color, 3)
            cv2.putText(panel, f"{label} frame {frame}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            panels.append(panel)
        joined = np.concatenate(panels, axis=1)
        if not cv2.imwrite(str(output / f"frame_{frame:06d}.jpg"), cv2.cvtColor(joined, cv2.COLOR_RGB2BGR)):
            raise OSError("Could not write review overlay")
