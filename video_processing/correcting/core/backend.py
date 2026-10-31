"""Pinned FoundationPose API with real observed RGB and monocular depth."""
from functools import lru_cache
import numpy as np
from .algorithm import Candidate, assess, depth_evidence

class NativeBackend:
    def __init__(self, native, export, depth, output, settings):
        self.native, self.export, self.depth = native, export, depth
        self.settings = settings
        self.K, w2c = native['camera_calibration'](native['load_edex'](export), 0)
        self.c2w = np.linalg.inv(w2c)
        (output / 'debug').mkdir(parents=True, exist_ok=True)
        self.estimator = native['_pose_estimator'](native['load_export_object_mesh'](export), output / 'debug')
        self.tf = native['_tensor_to_numpy'](self.estimator.get_tf_to_centered_mesh())
        self.inverse_tf = np.linalg.inv(self.tf)
        self.registration_calls = 0
        self.tracking_calls = 0
        self.rgb_only_tracking_calls = 0
        self.registration_skips = []

    @lru_cache(maxsize=3)
    def frame(self, index):
        return self.native['_read_foundationpose_frame'](self.export, self.depth, 0, f'{index:06d}', self.K)[1:]

    def track(self, frame, camera_pose):
        rgb, depth, _, _, K = self.frame(frame)
        torch = self.native['torch']
        self.estimator.pose_last = torch.as_tensor(camera_pose @ self.inverse_tf, device='cuda', dtype=torch.float32)
        result = self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=self.settings['tracking_iteration'])
        self.tracking_calls += 1
        return np.asarray(result, dtype=np.float32).reshape(4, 4)

    def rendered_depth(self, poses, K, shape):
        n = self.native; torch, dr, utils = n['torch'], n['dr'], n['Utils']
        estimator = self.estimator
        device = estimator.mesh_tensors['pos'].device
        projection = torch.as_tensor(utils.projection_matrix_from_intrinsics(K, height=shape[0], width=shape[1], znear=.001, zfar=100.), device=device, dtype=torch.float32)
        cv_to_gl = torch.as_tensor(utils.glcam_in_cvcam, device=device, dtype=torch.float32)
        vertices = utils.to_homo_torch(estimator.mesh_tensors['pos'])
        output = []
        with torch.no_grad():
            for begin in range(0, len(poses), 4):
                transform = torch.as_tensor(poses[begin:begin+4], device=device, dtype=torch.float32)
                camera = torch.einsum('bij,vj->bvi', transform, vertices)
                clip = torch.einsum('ij,bvj->bvi', projection @ cv_to_gl, camera).contiguous()
                raster, _ = dr.rasterize(estimator.glctx, clip, estimator.mesh_tensors['faces'], resolution=shape)
                depth, _ = dr.interpolate(camera[..., 2:3].contiguous(), raster, estimator.mesh_tensors['faces'])
                depth = torch.where(raster[..., 3:4] > 0, depth, torch.zeros_like(depth))
                output.extend(n['_tensor_to_numpy'](torch.flip(depth[..., 0], dims=[1])))
        return np.asarray(output)

    def measure(self, frame, camera_poses, origin, anchor_id=None, registered=False):
        _, depth, mask_h, mask_o, K = self.frame(frame)
        n = self.native; centered = np.asarray(camera_poses) @ self.inverse_tf
        _, _, _, metrics = n['_full_candidate_data'](self.estimator, centered, np.arange(len(centered)), np.arange(len(centered)), K, mask_h, mask_o)
        scaled_K, shape = n['_scaled_image_grid'](K, mask_o.shape, 768)
        depths = self.rendered_depth(centered, scaled_K, shape)
        cv2 = n['cv2']
        observed = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mh, mo = n['_resize_mask'](mask_h, shape), n['_resize_mask'](mask_o, shape)
        result = []
        for i, pose in enumerate(camera_poses):
            values = n['_selected_metric_values'](metrics, i)
            values.update(depth_evidence(depths[i], observed, depths[i] > 0, mh, mo))
            eligible, score, reasons = assess(values, self.settings)
            result.append(Candidate(np.asarray(pose, dtype=np.float32).copy(), origin, values, eligible, score, reasons, anchor_id, i, registered))
        return result

