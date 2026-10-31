"""Audited MHR to 77-joint SOMA fit; calibration is handled separately."""
import importlib.metadata
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from lib_mhr.mhr_layer import MHRParametricBackend
from soma import SOMALayer
from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec
from soma.pose_inversion import PoseInversion
from soma.units import Unit

ASSET_REVISION = "466879a83d57eabf3d875ded2d869f2075f90348"

class Converter:
    def __init__(self, weights_root):
        if importlib.metadata.version("py-soma-x") != "0.2.1":
            raise ValueError("This adapter is checked against py-soma-x 0.2.1")
        self.backend = MHRParametricBackend(mhr_assets_root=str(weights_root / "sam3d_body"), device="cuda")
        self.soma = SOMALayer(data_root=str(weights_root / "soma" / ASSET_REVISION), identity_model_type="mhr",
                              device="cuda", mode="warp", output_unit=Unit.CENTIMETERS,
                              enable_procedural_transforms=False)
        self.inv = PoseInversion(self.soma, low_lod=True)
        self.names = np.asarray(list(self.soma.public_joint_names)[1:])

    @torch.no_grad()
    def convert(self, params):
        context = self.backend._vertices_context(params, detach_fixed=True)
        trans, body, shape, scale = self.backend._mutable_vertices_inputs(params, context)
        verts, resolved = context.head.mhr_forward(global_trans=torch.zeros_like(trans),
            global_rot=context.global_rot, body_pose_params=body, hand_pose_params=context.hand,
            scale_params=scale, shape_params=shape, expr_params=context.face,
            return_keypoints=False, return_joint_coords=False, return_model_params=True, return_joint_rotations=False)
        target_cm = (verts + trans[:, None, :] * context.flip) * 100.0
        if not torch.isfinite(target_cm).all() or not torch.isfinite(resolved).all():
            raise ValueError("Non-finite MHR decode")
        identity_kwargs = {"bone_length_flexibles": resolved[:, 130:136]}
        self.inv.prepare_identity(shape, resolved[:, 136:], kwargs=identity_kwargs)
        fit = self.inv.fit(target_cm, body_iters=2, finger_iters=0, full_iters=1,
                           lie_iters=3, lie_lambda=.1, leaf_weight=1., batch_size=None)
        self.soma.prepare_identity(shape, resolved[:, 136:], kwargs=identity_kwargs)
        relative = remove_joint_orient_local(fit["rotations"], self.soma._t_pose_orient,
                                            self.soma._t_pose_orient_parent_T)
        poses = matrix_to_rotvec(relative.reshape(-1, 3, 3)).reshape(len(trans), 78, 3)[:, 1:].contiguous()
        output = self.soma.pose(poses, fit["root_translation"], fk_only=True)
        # Audit the actual rotvec round-trip, rather than reporting only the
        # optimizer's internal vertex error before NVIDIA's output conversion.
        low = self.inv.soma.pose(poses, fit["root_translation"], apply_correctives=False)
        error_m = torch.linalg.vector_norm(low.vertices - self.inv.transfer_to_soma(target_cm), dim=-1) / 100.0
        transforms = output.transforms[:, 1:].cpu().numpy()
        r = np.diag([1., -1., -1.]) @ transforms[..., :3, :3]
        orthogonality = np.max(np.abs(r.swapaxes(-1, -2) @ r - np.eye(3)))
        if not np.isfinite(transforms).all() or orthogonality > 2e-5 or not torch.isfinite(error_m).all():
            raise ValueError(f"Invalid SOMA forward result; orthogonality={orthogonality}")
        q = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_quat().reshape(len(trans), 77, 4)
        def cpu(t):
            return t.detach().cpu().numpy()
        return {"positions": transforms[..., :3, 3] * [1., -1., -1.] / 100.0,
                "rotations_wxyz": q[..., [3, 0, 1, 2]], "poses": cpu(poses),
                "transl": cpu(fit["root_translation"]) / 100.0,
                "identity_coeffs": cpu(shape), "scale_params": cpu(resolved[:, 136:]),
                "bone_length_flexibles": cpu(resolved[:, 130:136]),
                "fit_error_mean_m": cpu(error_m.mean(dim=1)),
                "fit_error_p95_m": cpu(torch.quantile(error_m, .95, dim=1)),
                "fit_error_max_m": cpu(error_m.max(dim=1).values)}

