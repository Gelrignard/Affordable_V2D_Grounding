"""Small URDF FK reader for portable reference export (no simulator dependency)."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def vector(text, default):
    return np.asarray([float(x) for x in (text or default).split()])


class URDFKinematics:
    def __init__(self, path):
        robot = ET.parse(path).getroot()
        self.links = [node.attrib["name"] for node in robot.findall("link")]
        self.joints = []
        children = set()
        for node in robot.findall("joint"):
            origin = node.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = vector(origin.get("xyz"), "0 0 0")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", vector(origin.get("rpy"), "0 0 0")
                ).as_matrix()
            axis = node.find("axis")
            parent, child = (
                node.find("parent").get("link"),
                node.find("child").get("link"),
            )
            mimic = node.find("mimic")
            self.joints.append(
                (
                    node.get("name"),
                    node.get("type"),
                    parent,
                    child,
                    transform,
                    vector(axis.get("xyz") if axis is not None else None, "1 0 0"),
                    dict(mimic.attrib) if mimic is not None else None,
                )
            )
            children.add(child)
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        self.root = roots.pop()

    def poses(self, positions, root_position, root_wxyz, joint_names, links):
        n = len(positions)
        values = {name: positions[:, i] for i, name in enumerate(joint_names)}
        moving = {j[0] for j in self.joints if j[1] != "fixed" and j[6] is None}
        if moving != set(joint_names):
            raise ValueError(
                f"URDF joint names differ: {moving.symmetric_difference(joint_names)}"
            )
        root = np.tile(np.eye(4), (n, 1, 1))
        root[:, :3, 3] = root_position
        root[:, :3, :3] = Rotation.from_quat(root_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        transforms = {self.root: root}
        pending = list(self.joints)
        while pending:
            progress = False
            for joint in list(pending):
                name, kind, parent, child, origin, axis, mimic = joint
                if parent not in transforms:
                    continue
                motion = np.tile(np.eye(4), (n, 1, 1))
                if kind != "fixed":
                    angle = (
                        values[name]
                        if mimic is None
                        else (
                            values[mimic["joint"]] * float(mimic.get("multiplier", 1))
                            + float(mimic.get("offset", 0))
                        )
                    )
                    if kind in ("revolute", "continuous"):
                        motion[:, :3, :3] = Rotation.from_rotvec(
                            angle[:, None] * axis
                        ).as_matrix()
                    elif kind == "prismatic":
                        motion[:, :3, 3] = angle[:, None] * axis
                    else:
                        raise ValueError(f"unsupported URDF joint type {kind}")
                transforms[child] = transforms[parent] @ origin @ motion
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError("disconnected or cyclic URDF")
        result = []
        for link in links:
            xf = transforms[link]
            xyzw = Rotation.from_matrix(xf[:, :3, :3]).as_quat()
            result.append(
                np.concatenate((xf[:, :3, 3], xyzw[:, [3, 0, 1, 2]]), axis=-1)
            )
        return np.stack(result, axis=1)
