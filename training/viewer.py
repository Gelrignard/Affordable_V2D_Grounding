"""Serve a local SONIC wiring guide and real reference FK preview (no simulation)."""

import argparse
import json
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import RUNTIME_COMMIT
from .bundles import arrays, sha256
from .kinematics import URDFKinematics
from .prepare import load_job


def build_viewer(jobs, urdf, experiment, output):
    selection = json.loads(Path(experiment).read_text(encoding="utf-8"))
    kin = URDFKinematics(urdf)
    edges = [[kin.links.index(j[2]), kin.links.index(j[3])] for j in kin.joints]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {
        "runtime_commit": RUNTIME_COMMIT,
        "experiment": selection,
        "links": kin.links,
        "edges": edges,
        "sequences": [],
    }
    for tier in selection["tiers"]:
        for episode in selection["episodes"]:
            item = {
                "tier": tier,
                "episode": episode,
                "label": selection["episode_labels"][str(episode)],
                "variants": {},
            }
            for variant in selection["variants"]:
                folder = Path(jobs) / f"tier{tier}" / f"episode_{episode:06d}" / variant
                job = load_job(folder)
                ref = arrays(folder / "reference.npz")
                # Preview-only sampling; training and exported rollouts keep every source frame.
                ids = np.unique(
                    np.r_[
                        np.arange(0, job["source_frames"], 3), job["source_frames"] - 1
                    ]
                )
                poses = kin.poses(
                    ref["joint_positions"][ids],
                    ref["root_position"][ids],
                    ref["root_wxyz"][ids],
                    ref["joint_names"],
                    kin.links,
                )
                mesh = trimesh.load(
                    folder / "assets/object.glb", force="mesh", process=False
                )
                bounds = mesh.bounds
                corners = np.array(
                    [
                        [bounds[x, 0], bounds[y, 1], bounds[z, 2]]
                        for x in range(2)
                        for y in range(2)
                        for z in range(2)
                    ]
                )
                rotations = Rotation.from_quat(
                    ref["object_wxyz"][ids][:, [1, 2, 3, 0]]
                ).as_matrix()
                boxes = (
                    np.einsum("tij,kj->tki", rotations, corners)
                    + ref["object_position"][ids, None]
                )
                item["variants"][variant] = {
                    "points": poses[..., :3].round(5).tolist(),
                    "object_boxes": boxes.round(5).tolist(),
                    "object_path": ref["object_position"][ids].round(5).tolist(),
                    "observed": ref["object_observed_mask"][ids].astype(bool).tolist(),
                    "source_sha256": sha256(folder / "reference.npz"),
                }
                if variant == selection["variants"][0]:
                    item.update(
                        frames=ref["source_frame_indices"][ids].tolist(),
                        times=(ref["timestamps"][ids] - ref["timestamps"][0])
                        .round(6)
                        .tolist(),
                        source_frames=job["source_frames"],
                    )
            original = np.array(item["variants"]["original"]["points"])
            corrected = np.array(item["variants"]["corrected"]["points"])
            item["max_link_delta_mm"] = (
                (np.linalg.norm(original - corrected, axis=-1).max(axis=-1) * 1000)
                .round(2)
                .tolist()
            )
            data["sequences"].append(item)
    (output / "data.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(Path(__file__).parent / "web" / name, output / name)
    return output


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path not in (
            "/",
            "/index.html",
            "/app.js",
            "/style.css",
            "/data.json",
        ):
            self.send_error(404)
            return
        super().do_GET()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument(
        "--experiment",
        default=str(
            Path(__file__).resolve().parent / "configs/rl_pilot_two.json"
        ),
    )
    p.add_argument("--output", default="outputs/sonic_viewer")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()
    output = build_viewer(args.jobs, args.urdf, args.experiment, args.output)
    if args.build_only:
        print(output)
        return
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(ViewerHandler, directory=str(output))
    )
    print(
        f"SONIC wiring and reference preview: http://127.0.0.1:{args.port}", flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
