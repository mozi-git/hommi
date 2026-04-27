# From https://github.com/real-stanford/detached-umi-policy
from typing import Dict, Callable, Tuple, List, Optional
import sys
import os
import time
import click
import numpy as np
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
import json
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import dill
import hydra
import zmq
from torch.utils.data import DataLoader

# Compatibility shims for packages that still reference deprecated numpy aliases.
_np_aliases = {
    'float_': np.float64,
    'complex_': np.complex128,
}
for alias, target in _np_aliases.items():
    if not hasattr(np, alias):
        setattr(np, alias, target)

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from inference_utils import get_pointmap_from_stereo, rectify_image_batch
from diffusion_policy.common.pytorch_util import dict_apply
from hommi.common.cv_util import get_image_transform_with_border
from diffusion_policy.common.pose_repr_util import (
    convert_pose_mat_rep
)
from umi.common.pose_util import (
    pose_to_mat, mat_to_pose,
    mat_to_pose10d, pose10d_to_mat)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deps'))
from FoundationStereo.core.foundation_stereo import FoundationStereo
import omegaconf
import traceback

def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)


@dataclass
class DebugAttnRequest:
    output_dir: Path
    pointmaps: Dict[str, np.ndarray]
    pointmap_masks: Dict[str, np.ndarray]
    pointmaps_ref: Dict[str, np.ndarray]
    pointmap_rgbs: Dict[str, np.ndarray]
    gripper_poses: Dict[str, np.ndarray]
    lookat_point: Optional[np.ndarray]
    vit_viz: Dict[str, np.ndarray]
    pointmap_viz: Dict[str, np.ndarray]


def _sanitize_key(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key)


def _normalize_image_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8, copy=False)


def _sample_axis_points(pose: np.ndarray, axis_length: float, samples: int) -> tuple[np.ndarray, np.ndarray]:
    origin = pose[:3, 3]
    axes = pose[:3, :3]
    ts = np.linspace(0.0, axis_length, samples, dtype=np.float32)
    axis_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    points_list = []
    colors_list = []
    for axis_idx in range(3):
        direction = axes[:, axis_idx]
        pts = origin[None, :] + ts[:, None] * direction[None, :]
        points_list.append(pts)
        colors_list.append(np.tile(axis_colors[axis_idx], (samples, 1)))
    return np.concatenate(points_list, axis=0), np.concatenate(colors_list, axis=0)


def _sample_lookat_marker(
    lookat_point: np.ndarray,
    radius: float,
    samples: int,
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(-radius, radius, samples, dtype=np.float32)
    points_list = []
    colors_list = []
    for axis_idx in range(3):
        direction = np.zeros(3, dtype=np.float32)
        direction[axis_idx] = 1.0
        pts = lookat_point[None, :] + ts[:, None] * direction[None, :]
        points_list.append(pts)
        colors_list.append(np.tile(color, (samples, 1)))
    return np.concatenate(points_list, axis=0), np.concatenate(colors_list, axis=0)


def _save_debug_attention(request: DebugAttnRequest) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[policy] Unable to save attention debug assets (cv2 missing): {exc}")
        return

    request.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from hommi.train_network.model.vision_3d.dit_obs3d_encoder import save_attention_data
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[policy] Unable to import save_attention_data: {exc}")
        save_attention_data = None

    def _save_pointmap_set(pointmap_set: Dict[str, np.ndarray], suffix: str) -> None:
        if save_attention_data is None:
            return
        for key, pointmaps in pointmap_set.items():
            safe_key = _sanitize_key(key)
            timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_key}_{suffix}"
            mask = request.pointmap_masks.get(key)
            points = np.asarray(pointmaps)[-1].reshape(-1, 3)
            rgb = request.pointmap_rgbs.get(key)
            if rgb is not None:
                rgb_frame = np.asarray(rgb)[-1]
                if rgb_frame.ndim == 3:
                    rgb_frame = np.transpose(rgb_frame, (1, 2, 0))
                colors = rgb_frame.reshape(-1, 3)
                if np.issubdtype(colors.dtype, np.floating):
                    colors = np.clip(colors, 0.0, 1.0)
                else:
                    colors = np.clip(colors, 0, 255).astype(np.float32) / 255.0
            else:
                colors = np.full(points.shape, 0.5, dtype=np.float32)
            if mask is not None:
                mask_frame = np.asarray(mask)[-1]
                if mask_frame.ndim == 3:
                    mask_frame = mask_frame[0]
                mask_flat = mask_frame.reshape(-1)
                points = points[mask_flat]
                colors = colors[mask_flat]
            finite_mask = np.all(np.isfinite(points), axis=1)
            points = points[finite_mask]
            colors = colors[finite_mask]
            if suffix == "ref" and (request.gripper_poses or request.lookat_point is not None):
                extra_points = []
                extra_colors = []
                for pose in request.gripper_poses.values():
                    pts, cols = _sample_axis_points(pose, axis_length=0.08, samples=10)
                    extra_points.append(pts)
                    extra_colors.append(cols)
                if request.lookat_point is not None:
                    pts, cols = _sample_lookat_marker(
                        request.lookat_point.astype(np.float32, copy=False),
                        radius=0.02,
                        samples=10,
                        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                    extra_points.append(pts)
                    extra_colors.append(cols)
                if extra_points:
                    points = np.concatenate([points] + extra_points, axis=0)
                    colors = np.concatenate([colors] + extra_colors, axis=0)
            save_attention_data(points, colors, str(request.output_dir), timestamp)

    _save_pointmap_set(request.pointmaps, "raw")
    _save_pointmap_set(request.pointmaps_ref, "ref")

    vit_dir = request.output_dir / "vit_attention"
    vit_dir.mkdir(parents=True, exist_ok=True)
    for key, img in request.vit_viz.items():
        try:
            image = _normalize_image_array(img)
        except ValueError as exc:
            print(f"[policy] Skipping vit attention image '{key}': {exc}")
            continue
        output_path = vit_dir / f"{_sanitize_key(key)}.png"
        bgr = image[..., ::-1]
        cv2.imwrite(str(output_path), bgr)

    pcd_dir = request.output_dir / "pointmap_attention"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    for key, img in request.pointmap_viz.items():
        try:
            image = _normalize_image_array(img)
        except ValueError as exc:
            print(f"[policy] Skipping pointmap attention image '{key}': {exc}")
            continue
        output_path = pcd_dir / f"{_sanitize_key(key)}.png"
        bgr = image[..., ::-1]
        cv2.imwrite(str(output_path), bgr)


def _start_debug_attn_worker(
    max_queue: int = 4,
) -> tuple["queue.Queue[Optional[DebugAttnRequest]]", threading.Thread, threading.Event]:
    debug_queue: "queue.Queue[Optional[DebugAttnRequest]]" = queue.Queue(maxsize=max_queue)
    stop_event = threading.Event()

    def _worker() -> None:
        while True:
            if stop_event.is_set() and debug_queue.empty():
                break
            try:
                request = debug_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                if stop_event.is_set():
                    break
                continue
            try:
                _save_debug_attention(request)
            except Exception as exc:  # pragma: no cover - best effort saver
                print(f"[policy] Failed to save attention debug assets: {exc}")

    thread = threading.Thread(target=_worker, name="policy-attn-writer", daemon=True)
    thread.start()
    return debug_queue, thread, stop_event


def _enqueue_debug_attn(
    debug_queue: "queue.Queue[Optional[DebugAttnRequest]]",
    debug_counter: List[int],
    output_root: Path,
    pointmaps: Dict[str, np.ndarray],
    pointmap_masks: Dict[str, np.ndarray],
    pointmaps_ref: Dict[str, np.ndarray],
    pointmap_rgbs: Dict[str, np.ndarray],
    gripper_poses: Dict[str, np.ndarray],
    lookat_point: Optional[np.ndarray],
    vit_viz: Dict[str, np.ndarray],
    pointmap_viz: Dict[str, np.ndarray],
) -> None:
    if not pointmaps and not vit_viz and not pointmap_viz:
        return
    if debug_queue.full():
        print("[policy] Attention debug queue full; dropping batch.")
        return
    debug_counter[0] += 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{stamp}_{debug_counter[0]:03d}"
    debug_queue.put(
        DebugAttnRequest(
            output_dir=output_dir,
            pointmaps=dict(pointmaps),
            pointmap_masks=dict(pointmap_masks),
            pointmaps_ref=dict(pointmaps_ref),
            pointmap_rgbs=dict(pointmap_rgbs),
            gripper_poses=dict(gripper_poses),
            lookat_point=None if lookat_point is None else np.asarray(lookat_point, dtype=np.float32),
            vit_viz=dict(vit_viz),
            pointmap_viz=dict(pointmap_viz),
        )
    )


class PolicyInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int, device: str, frequency: float, action_horizon: int = 8, use_gt_action: bool = False, pointmap_scale: float = 1.0, intrinsic_file: str = '../rainbow/K_head.txt', fs_ckpt_dir: str = '.../deps/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth', vis_attn: bool = False, debug_save_attn: bool = False, reference_frame: str = None, head_stereo_crop=None, stereo_rectify: bool = True, fisheye_calibration_json: Optional[str] = None, stereo_calibration_json: Optional[str] = None, save_rectified: bool = False, debug_stereo_k: bool = False):
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        self.cfg.task.eval_image_transforms.transforms['camera_left_main_rgb'][0].ratio=1.0
        self.cfg.task.eval_image_transforms.transforms['camera_right_main_rgb'][0].ratio=1.0
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
        with open(cfg_path, 'w') as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"[policy] Exported config to {cfg_path}")
        print(f"[policy] Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.model._target_}, obs_encoder: {self.cfg.model.obs_encoder._target_}")

        assert pointmap_scale <= 1.0, "pointmap_scale must be <= 1.0"
        self.pointmap_scale = pointmap_scale
        self.stereo_rectify = bool(stereo_rectify)

        # load baseline from config
        with open(intrinsic_file, 'r') as f:
            lines = f.readlines()
            self.baseline = float(lines[1]) if len(lines) > 1 else 0.0
        self.dist_model = "pinhole"
        self.calib_dim = None
        self.dist_coeffs = None
        self.K_right = None
        self.dist_coeffs_right = None
        self.rect_R1 = None
        self.rect_R2 = None
        self.rect_P1 = None
        self.rect_P2 = None
        if stereo_calibration_json:
            with open(stereo_calibration_json, "r") as f:
                data = json.load(f)
            self.calib_dim = tuple(data["DIM"])
            self.K = np.array(data["left"]["K"], dtype=np.float32)
            self.dist_coeffs = np.array(data["left"]["D"], dtype=np.float32).reshape(-1)
            self.K_right = np.array(data["right"]["K"], dtype=np.float32)
            self.dist_coeffs_right = np.array(data["right"]["D"], dtype=np.float32).reshape(-1)
            self.rect_R1 = np.array(data["rectify"]["R1"], dtype=np.float32)
            self.rect_R2 = np.array(data["rectify"]["R2"], dtype=np.float32)
            self.rect_P1 = np.array(data["rectify"]["P1"], dtype=np.float32)
            self.rect_P2 = np.array(data["rectify"]["P2"], dtype=np.float32)
            T = np.array(data["stereo"]["T"], dtype=np.float32)
            self.baseline = float(np.linalg.norm(T))
            self.dist_model = "fisheye"
        elif fisheye_calibration_json:
            with open(fisheye_calibration_json, "r") as f:
                data = json.load(f)
            self.calib_dim = tuple(data["DIM"])
            self.K = np.array(data["K"], dtype=np.float32)
            self.dist_coeffs = np.array(data["D"], dtype=np.float32).reshape(-1)
            self.dist_model = "fisheye"
        else:
            self.K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            if len(lines) >= 3 and lines[2].strip():
                self.dist_coeffs = np.array(list(map(float, lines[2].rstrip().split()))).astype(np.float32)
        self.debug_stereo_k = bool(debug_stereo_k)
        dataset_cfg = self.cfg.task.get('dataset', {})
        self.crop_pointcloud = bool(dataset_cfg.get('crop_pointcloud', False))
        self.depth_range = tuple(dataset_cfg.get('depth_range', (0.1, 1.5)))
        self.gripper_length = float(dataset_cfg.get('gripper_length', 0.13) or 0.13)
        self.head_stereo_crop = self._normalize_crop_ratio(
            head_stereo_crop
            if head_stereo_crop is not None
            else self.cfg.task.get('head_stereo_crop', dataset_cfg.get('head_stereo_crop', None))
        )
        if self.head_stereo_crop is not None:
            print(f"[policy] Using head stereo center crop ratio (w,h): {self.head_stereo_crop}")
        self.get_class_start_time = time.monotonic()
        self.rectify_debug_dir = None
        if self.stereo_rectify and save_rectified:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.rectify_debug_dir = os.path.join(os.path.dirname(__file__), "runs", f"rectify_{stamp}")
            os.makedirs(self.rectify_debug_dir, exist_ok=True)

        # load foundation stereo model
        if fs_ckpt_dir is not None and os.path.exists(fs_ckpt_dir):
            fs_cfg = omegaconf.OmegaConf.load(f'{os.path.dirname(fs_ckpt_dir)}/cfg.yaml')
            if 'vit_size' not in fs_cfg:
                fs_cfg['vit_size'] = 'vitl'
            fs_model = FoundationStereo(omegaconf.OmegaConf.create(fs_cfg))
            fs_ckpt = torch.load(fs_ckpt_dir, weights_only=False)
            fs_model.load_state_dict(fs_ckpt['model'])
            fs_model.cuda()
            fs_model.eval()
            self.fs_model = fs_model
            print(f"[policy] Loaded FoundationStereo model from {fs_ckpt_dir}")
        else:
            print(f"[policy] Not loading FoundationStereo model from {fs_ckpt_dir}, since it does not exist.")
            self.fs_model = None

        # initialize image and pointmap transforms
        self.transforms = dict()

        if not use_gt_action:
            cls = hydra.utils.get_class(self.cfg._target_)
            self.workspace = cls(self.cfg)
            self.workspace: BaseWorkspace
            # Newer checkpoints may contain additional modules (e.g. obs normalizers)
            # that are not present in the current runtime version of the policy.
            # Use strict=False so load_state_dict ignores these mismatches instead of erroring.
            self.workspace.load_payload(
                payload,
                exclude_keys=['optimizer'],
                include_keys=None,
                strict=False,
            )

            self.policy:BaseImagePolicy = self.workspace.model
            if self.cfg.training.use_ema:
                self.policy = self.workspace.ema_model
                print("[policy] Using EMA model")
            self.policy.num_inference_steps = 16
            if hasattr(self.policy, "set_normalizer"):
                if hasattr(self.policy, "normalizer"):
                    self.policy.set_normalizer(self.policy.normalizer)
                elif hasattr(self.workspace, "normalizer"):
                    self.policy.set_normalizer(self.workspace.normalizer)
            print(
                f"[policy] Model-side normalization enabled: {not getattr(self.policy, '_skip_model_side_normalization', False)}"
            )
            obs_encoder = getattr(self.policy, "obs_encoder", None)
            has_obs_norm = getattr(obs_encoder, "obs_normalizer", None) is not None if obs_encoder is not None else False
            print(f"[policy] Obs encoder normalizer set: {has_obs_norm}")
            if hasattr(self.policy.obs_encoder, "vis_attention"):
                self.policy.obs_encoder.vis_attention = bool(vis_attn)
            if hasattr(self.policy.obs_encoder, "save_attention_outputs"):
                self.policy.obs_encoder.save_attention_outputs = bool(vis_attn)

        self.shape_meta = self.cfg.task.shape_meta
        self.obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('[policy] obs_pose_rep', self.obs_pose_rep)
        print('[policy] action_pose_repr', self.action_pose_repr)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim" and not attr.get('is_language', False) and attr.get('in_replay_buffer', True):
                lowdim_keys.append(key)
        self.num_robot = 0
        self.id_to_side = dict()
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
                self.id_to_side[self.num_robot - 1] = key.split('_eef_pos')[0]
        self.action_indexing = dict()
        self.action_indexing_raw = dict()
        action_index = 0
        action_index_raw = 0
        for robot_idx in range(self.num_robot):
            for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width', 'lookatpoint']:
                if cat == 'lookatpoint' and self.id_to_side[robot_idx] == 'gripper_head':
                    key = 'camera_head_lookatpoint'
                else:
                    key = f'{self.id_to_side[robot_idx]}_{cat}'
                if key in obs_shape_meta and obs_shape_meta[key].get('ignore_by_action', False) is not True:
                    self.action_indexing[key] = (action_index, action_index + obs_shape_meta[key]['shape'][-1])
                    if 'raw_shape' in obs_shape_meta[key]:
                        self.action_indexing_raw[key] = (action_index_raw, action_index_raw + obs_shape_meta[key]['raw_shape'][-1])
                        action_index_raw += obs_shape_meta[key]['raw_shape'][-1]
                    else:
                        self.action_indexing_raw[key] = (action_index_raw, action_index_raw + obs_shape_meta[key]['shape'][-1])
                        action_index_raw += obs_shape_meta[key]['shape'][-1]
                    action_index += obs_shape_meta[key]['shape'][-1]
        print(f'[policy] action indexing: {self.action_indexing}')
        print(f'[policy] action indexing raw: {self.action_indexing_raw}')
        self.device = torch.device(device)
        if not use_gt_action:
            self.policy.eval().to(self.device)
            self.policy.reset()
        self.ip = ip
        self.port = port
        self.frequency = frequency
        self.dt = 1.0 / self.frequency
        self.action_horizon = action_horizon
        self.reference_frame = reference_frame if reference_frame is not None else self.cfg.task.get('reference_frame', 'world')
        print(f'[policy] Using reference frame: {self.reference_frame}')
        # Convert to plain Python container for ZMQ clients (DictConfig would not satisfy isinstance(..., dict) checks)
        self.required_obs_keys = omegaconf.OmegaConf.to_container(obs_shape_meta, resolve=True)
        # self.reference_frame = 'left'
        self._debug_attn_queue: Optional["queue.Queue[Optional[DebugAttnRequest]]"] = None
        self._debug_attn_thread: Optional[threading.Thread] = None
        self._debug_attn_stop: Optional[threading.Event] = None
        self._debug_attn_counter = [0]
        self._debug_attn_dir = Path("./runs/policy_attn_debug")
        self._last_pointmaps: Dict[str, np.ndarray] = {}
        self._last_pointmap_masks: Dict[str, np.ndarray] = {}
        self._last_pointmap_rgbs: Dict[str, np.ndarray] = {}
        self._last_pointmaps_ref: Dict[str, np.ndarray] = {}
        if debug_save_attn:
            self._debug_attn_queue, self._debug_attn_thread, self._debug_attn_stop = _start_debug_attn_worker()

    def _get_debug_gripper_poses(self, env_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        base_pose_mat = None
        if self.reference_frame == 'world':
            head_tf_key = 'gripper_head_tf' if 'gripper_head_tf' in env_obs else ('head_tf' if 'head_tf' in env_obs else None)
            if head_tf_key is not None:
                base_pose_mat = np.asarray(env_obs[head_tf_key], dtype=float)[-1]
        else:
            base_prefix = f'gripper_{self.reference_frame}'
            base_tf_key = base_prefix + '_tf'
            if base_tf_key not in env_obs and self.reference_frame == 'head' and 'head_tf' in env_obs:
                base_tf_key = 'head_tf'
            if base_tf_key in env_obs:
                base_pose_mat = np.asarray(env_obs[base_tf_key], dtype=float)[-1]
        if base_pose_mat is None:
            return {}

        base_inv = np.linalg.inv(base_pose_mat)
        gripper_poses = {}
        for prefix in self.id_to_side.values():
            tf_key = f"{prefix}_tf"
            if tf_key not in env_obs:
                continue
            pose_mat = np.asarray(env_obs[tf_key], dtype=float)[-1]
            gripper_poses[prefix] = base_inv @ pose_mat
        return gripper_poses

    def _get_debug_lookat_point(
        self,
        raw_action: np.ndarray,
        env_obs: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        if 'camera_head_lookatpoint' not in self.action_indexing:
            return None
        start, end = self.action_indexing['camera_head_lookatpoint']
        if raw_action.shape[-1] < end:
            return None
        lookat_action = raw_action[..., start:end]
        if lookat_action.size == 0:
            return None
        lookat_point = np.asarray(lookat_action[0], dtype=np.float32)
        if self.reference_frame == 'world':
            head_tf_key = 'gripper_head_tf' if 'gripper_head_tf' in env_obs else ('head_tf' if 'head_tf' in env_obs else None)
            if head_tf_key is not None:
                head_pose_mat = np.asarray(env_obs[head_tf_key], dtype=float)[-1]
                lookat_h = np.concatenate([lookat_point, np.array([1.0], dtype=np.float32)], axis=0)
                lookat_point = (np.linalg.inv(head_pose_mat) @ lookat_h)[:3]
        return lookat_point

    def predict_action(self, obs_dict_np: dict):
        with torch.no_grad():
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()[:self.action_horizon, ...]
            # print(f"[policy] Predicted action {action}")
            del result
            del obs_dict
        return action

    def get_real_umi_obs_dict(
            self,
            env_obs: Dict[str, np.ndarray],
            tx_robot1_robot0: np.ndarray=None,
            episode_start_pose: List[np.ndarray]=None,
            ) -> Dict[str, np.ndarray]:
        obs_dict_np = dict()
        pointmap_masks: Dict[str, np.ndarray] = dict()
        debug_pointmaps: Dict[str, np.ndarray] = {}
        debug_pointmap_rgbs: Dict[str, np.ndarray] = {}
        debug_pointmaps_ref: Dict[str, np.ndarray] = {}
        # process non-pose
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            shape = attr.get('shape')
            if type == 'rgb':
                if key not in env_obs:
                    print(f"[policy] Missing RGB key '{key}' in observation; skipping.")
                    continue
                this_imgs_in = env_obs[key]
                if key in ("camera_head_main_rgb", "camera_head_main_right_rgb") and self.stereo_rectify and self.dist_coeffs is not None:
                    rect_R = self.rect_R1 if key == "camera_head_main_rgb" else self.rect_R2
                    rect_P = self.rect_P1 if key == "camera_head_main_rgb" else self.rect_P2
                    this_imgs_in, _ = rectify_image_batch(
                        imgs=this_imgs_in,
                        K=self.K,
                        dist_coeffs=self.dist_coeffs,
                        dist_model=self.dist_model,
                        baseline=self.baseline,
                        calib_dim=self.calib_dim,
                        rect_R=rect_R,
                        rect_P=rect_P,
                    )
                if self.head_stereo_crop is not None and key in ('camera_head_main_rgb', 'camera_head_main_right_rgb'):
                    this_imgs_in = self._center_crop_rgb(
                        this_imgs_in,
                        self.head_stereo_crop
                    )
                t,hi,wi,ci = this_imgs_in.shape
                co,ho,wo = shape
                assert ci == co
                out_imgs = this_imgs_in
                if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                    # tf = get_image_transform(
                    #     input_res=(wi,hi),
                    #     output_res=(wo,ho),
                    #     bgr_to_rgb=False)
                    if self.transforms.get(key) is None:
                        self.transforms[key] = get_image_transform_with_border(
                            in_res=(wi, hi),
                            out_res=(wo, ho),
                            bgr_to_rgb=True)
                    tf = self.transforms[key]
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                    if this_imgs_in.dtype == np.uint8:
                        out_imgs = out_imgs.astype(np.float32) / 255
                # THWC to TCHW
                obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
            elif type == 'pointmap':
                # generate pointmap from stereo images
                left_key = key.replace('_pointmap', '_main_rgb')
                right_key = key.replace('_pointmap', '_main_right_rgb')
                if left_key not in env_obs or right_key not in env_obs:
                    print(f"[policy] Missing stereo keys '{left_key}'/'{right_key}' for pointmap '{key}'; skipping.")
                    continue
                imgs_left = env_obs[left_key]    # [T, H, W, 3]
                imgs_right = env_obs[right_key] # [T, H, W, 3]
                K_for_stereo = self.K
                use_rectify = self.stereo_rectify and (self.dist_coeffs is not None)
                if self.head_stereo_crop is not None and (not use_rectify) and ('camera_head' in left_key or 'camera_head' in right_key):
                    imgs_left, imgs_right, K_for_stereo = self._center_crop_stereo_pair(
                        imgs_left,
                        imgs_right,
                        self.head_stereo_crop,
                        self.K
                    )
                pointmaps = get_pointmap_from_stereo(
                    model=self.fs_model,
                    imgs_left=imgs_left,
                    imgs_right=imgs_right,
                    K=K_for_stereo,
                    baseline=self.baseline,
                    scale=self.pointmap_scale,
                    rectify=use_rectify,
                    dist_model=self.dist_model if use_rectify else "pinhole",
                    dist_coeffs=self.dist_coeffs if use_rectify else None,
                    dist_coeffs_right=self.dist_coeffs_right if use_rectify else None,
                    K_right=self.K_right if use_rectify else None,
                    debug_dir=self.rectify_debug_dir if use_rectify else None,
                    calib_dim=self.calib_dim,
                    debug_print=self.debug_stereo_k,
                    rect_R1=self.rect_R1 if use_rectify else None,
                    rect_R2=self.rect_R2 if use_rectify else None,
                    rect_P1=self.rect_P1 if use_rectify else None,
                    rect_P2=self.rect_P2 if use_rectify else None,
                )   # [T, H, W, 3]
                if self.head_stereo_crop is not None and use_rectify and ('camera_head' in left_key or 'camera_head' in right_key):
                    pointmaps = self._center_crop_rgb(pointmaps, self.head_stereo_crop)
                t, hi, wi, ci = pointmaps.shape
                co, ho, wo = shape
                assert ci == co
                if (ho != hi) or (wo != wi):
                    if self.transforms.get(key) is None:
                        self.transforms[key] = get_image_transform_with_border(
                            in_res=(wi, hi),
                            out_res=(wo, ho),
                            mode='pointmap',)
                    tf = self.transforms[key]
                    pointmaps = np.stack([tf(x) for x in pointmaps])
                # optional depth and gripper cropping mask in head frame
                mask = np.ones((t, 1, pointmaps.shape[1], pointmaps.shape[2]), dtype=bool)
                if self.crop_pointcloud:
                    head_depth = pointmaps[..., 2]  # T, H, W
                    mask &= (head_depth[:, None, :, :] > self.depth_range[0]) & (head_depth[:, None, :, :] < self.depth_range[1])
                    head_tf_key = 'gripper_head_tf' if 'gripper_head_tf' in env_obs else ('head_tf' if 'head_tf' in env_obs else None)
                    if head_tf_key is not None:
                        head_pose_mat = np.asarray(env_obs[head_tf_key], dtype=float)
                        points_head = np.moveaxis(pointmaps, -1, 1)  # T,3,H,W
                        points_homogeneous = np.concatenate([
                            points_head,
                            np.ones(points_head.shape[:1] + (1,) + points_head.shape[2:], dtype=points_head.dtype)
                        ], axis=1)  # T,4,H,W
                        for side in ['left', 'right']:
                            side_tf_key = f'gripper_{side}_tf'
                            if side_tf_key not in env_obs:
                                continue
                            side_pose_mat = np.asarray(env_obs[side_tf_key], dtype=float)
                            curr_t = min(t, side_pose_mat.shape[0], head_pose_mat.shape[0])
                            if curr_t == 0:
                                continue
                            head_in_gripper_mat = np.stack(
                                [np.linalg.inv(side_pose_mat[idx]) @ head_pose_mat[idx] for idx in range(curr_t)],
                                axis=0)
                            points_in_gripper = np.einsum('tij,tjkm->tikm', head_in_gripper_mat, points_homogeneous[:curr_t])
                            mask[:curr_t] &= (points_in_gripper[:, 2:3, :, :] > -self.gripper_length)
                pointmap_masks[key] = mask
                obs_dict_np[key] = np.moveaxis(pointmaps, -1, 1)  # TCHW
                obs_dict_np[f'{key}_mask'] = mask.astype(bool)
                debug_pointmaps[key] = pointmaps.copy()
                rgb_for_pointmap = obs_dict_np.get(left_key)
                if rgb_for_pointmap is not None:
                    debug_pointmap_rgbs[key] = np.asarray(rgb_for_pointmap).copy()
            elif type == 'low_dim' and ('eef' not in key) and ('lookatpoint' not in key):
                this_data_in = env_obs[key]
                obs_dict_np[key] = this_data_in

        if self.reference_frame == 'world':
            pose_mats = {}
            for robot_prefix in self.id_to_side.values():
                tf_key = robot_prefix + '_tf'
                if tf_key not in env_obs:
                    continue
                pose_mats[robot_prefix] = np.asarray(env_obs[tf_key], dtype=float)

            # generate relative pose
            for robot_prefix, pose_mat in pose_mats.items():

                # solve reltaive obs
                obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=pose_mat[-1],
                    pose_rep=self.obs_pose_rep,
                    backward=False)

                obs_pose = mat_to_pose10d(obs_pose_mat)
                obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
                obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]

            # generate pose relative to other robot
            n_robots = len(self.id_to_side)
            for robot_id in range(n_robots):
                # convert pose to mat
                tx_robota_tcpa = pose_mats[self.id_to_side[robot_id]]
                for other_robot_id in range(n_robots):
                    if robot_id == other_robot_id:
                        continue
                    if self.id_to_side[robot_id] + '_eef_pos_wrt_' + self.id_to_side[other_robot_id].split("_")[-1] not in obs_shape_meta:
                        continue
                    tx_robotb_tcpb = pose_mats[self.id_to_side[other_robot_id]]
                    if tx_robot1_robot0 is not None:    # this should be passed in None since the poses are already in the same frame (are from the same robot)
                        tx_robota_robotb = tx_robot1_robot0
                        if robot_id == 0:
                            tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
                        tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb
                    else:
                        tx_robota_tcpb = tx_robotb_tcpb
                    rel_obs_pose_mat = convert_pose_mat_rep(
                        tx_robota_tcpa,
                        base_pose_mat=tx_robota_tcpb[-1],
                        pose_rep='relative',
                        backward=False)
                    rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                    obs_dict_np[f'{self.id_to_side[robot_id]}_eef_pos_wrt_{self.id_to_side[other_robot_id].split("_")[-1]}'] = rel_obs_pose[:,:3]
                    obs_dict_np[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle_wrt_{self.id_to_side[other_robot_id].split("_")[-1]}'] = rel_obs_pose[:,3:]

            # generate relative pose with respect to episode start
            if episode_start_pose is not None:
                for robot_id in range(n_robots):
                    pose_mat = pose_mats[self.id_to_side[robot_id]]

                    # get start pose
                    start_pose = episode_start_pose[robot_id]
                    start_pose_mat = pose_to_mat(start_pose)
                    rel_obs_pose_mat = convert_pose_mat_rep(
                        pose_mat,
                        base_pose_mat=start_pose_mat,
                        pose_rep='relative',
                        backward=False)

                    rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                    # obs_dict_np[f'{robot_id_to_side[robot_id]}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
                    obs_dict_np[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]
            debug_pointmaps_ref = dict(debug_pointmaps)
        else:
            print(f'using reference frame: {self.reference_frame}')
            assert self.reference_frame == 'left' or self.reference_frame == 'right' or self.reference_frame == 'head', f'unsupported reference frame {self.reference_frame}'
            # use the reference gripper pose as the base frame
            base_prefix = f'gripper_{self.reference_frame}'
            base_tf_key = base_prefix + '_tf'
            if base_tf_key not in env_obs:
                print(f"[policy] Missing base transform '{base_tf_key}' in observation; skipping pose transform.")
                return obs_dict_np
            base_pose_seq = np.asarray(env_obs[base_tf_key], dtype=float)
            base_pose_mat = base_pose_seq[-1]

            # transform head-based modalities (pointmap / lookatpoint) into the base frame
            pointmap_keys = [
                key for key, attr in obs_shape_meta.items()
                if attr.get('type') == 'pointmap'
            ]
            head_tf_key = 'gripper_head_tf' if 'gripper_head_tf' in env_obs else ('head_tf' if 'head_tf' in env_obs else None)
            if (len(pointmap_keys) > 0 or 'camera_head_lookatpoint' in obs_dict_np) and head_tf_key is not None:
                head_pose_mat = np.asarray(env_obs[head_tf_key], dtype=float)
                ref_head_pose_mat = convert_pose_mat_rep(
                    head_pose_mat,
                    base_pose_mat=base_pose_mat,
                    pose_rep='relative',
                    backward=False)
                for key in pointmap_keys:
                    if key not in obs_dict_np:
                        continue
                    head_points = obs_dict_np[key]    # (T, 3, H, W)
                    points_homogeneous = np.concatenate([
                        head_points,
                        np.ones(head_points.shape[:1] + (1,) + head_points.shape[2:], dtype=head_points.dtype)
                    ], axis=1)   # (T, 4, H, W)
                    points_transformed = np.einsum('tij,tjkm->tikm', ref_head_pose_mat, points_homogeneous)
                    obs_dict_np[key] = points_transformed[:, :3, :, :]
                    debug_pointmaps_ref[key] = np.moveaxis(obs_dict_np[key], 1, -1)
                if 'camera_head_lookatpoint' in obs_dict_np:
                    lookatpoint_obs = obs_dict_np['camera_head_lookatpoint']
                    lookatpoint_obs_homogeneous = np.concatenate([
                        lookatpoint_obs,
                        np.ones((lookatpoint_obs.shape[0], 1), dtype=lookatpoint_obs.dtype)
                    ], axis=-1)  # (T, 4)
                    lookatpoint_obs_transformed = np.einsum('tij,tj->ti', ref_head_pose_mat, lookatpoint_obs_homogeneous)
                    obs_dict_np['camera_head_lookatpoint'] = lookatpoint_obs_transformed[:, :3]
                    print(f'[policy] transformed camera_head_lookatpoint into {self.reference_frame} frame')

            # now transform all eef poses into the base frame
            for robot_idx in range(len(self.id_to_side)):
                prefix = self.id_to_side[robot_idx]
                tf_key = prefix + '_tf'
                if tf_key not in env_obs:
                    continue
                pose_mat = np.asarray(env_obs[tf_key], dtype=float)
                ref_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=base_pose_mat,
                    pose_rep='relative',
                    backward=False)
                obs_pose = mat_to_pose10d(ref_obs_pose_mat)
                obs_dict_np[f'{prefix}_eef_pos'] = obs_pose[:, :3]
                obs_dict_np[f'{prefix}_eef_rot_axis_angle'] = obs_pose[:, 3:]
        self._last_pointmaps = debug_pointmaps
        self._last_pointmap_masks = pointmap_masks
        self._last_pointmap_rgbs = debug_pointmap_rgbs
        self._last_pointmaps_ref = debug_pointmaps_ref
        return obs_dict_np

    @staticmethod
    def _normalize_crop_ratio(crop):
        if crop is None:
            return None
        if not isinstance(crop, (list, tuple, dict, str)):
            try:
                crop = omegaconf.OmegaConf.to_container(crop, resolve=True)
            except Exception:
                print(f"[policy] Invalid head_stereo_crop config '{crop}', expected list/tuple/dict/str")
                return None

        def to_ratio(value):
            if isinstance(value, str):
                value = value.strip()
                if value.endswith('%'):
                    value = value[:-1]
                    return float(value) / 100.0
            return float(value)

        ratios = None
        if isinstance(crop, str):
            parts = [p.strip() for p in crop.split(',') if p.strip() != ""]
            if len(parts) == 1:
                ratio = to_ratio(parts[0])
                ratios = (ratio, ratio)
            elif len(parts) == 2:
                ratios = (to_ratio(parts[0]), to_ratio(parts[1]))
            else:
                print(f"[policy] Invalid head_stereo_crop string '{crop}', expected 'ratio' or 'ratio_w,ratio_h'")
                return None
        elif isinstance(crop, dict):
            if 'ratio' in crop:
                ratio = to_ratio(crop['ratio'])
                ratios = (ratio, ratio)
            elif 'width' in crop and 'height' in crop:
                ratios = (to_ratio(crop['width']), to_ratio(crop['height']))
            else:
                print(f"[policy] Invalid head_stereo_crop dict '{crop}', expected keys ratio or width/height")
                return None
        else:
            if len(crop) == 1:
                ratio = to_ratio(crop[0])
                ratios = (ratio, ratio)
            elif len(crop) == 2:
                ratios = (to_ratio(crop[0]), to_ratio(crop[1]))
            else:
                print(f"[policy] Invalid head_stereo_crop list '{crop}', expected [ratio] or [ratio_w, ratio_h]")
                return None

        try:
            ratio_w, ratio_h = float(ratios[0]), float(ratios[1])
        except Exception:
            print(f"[policy] Invalid head_stereo_crop values '{ratios}', expected floats")
            return None
        if not (0.0 < ratio_w <= 1.0 and 0.0 < ratio_h <= 1.0):
            print(f"[policy] Invalid head_stereo_crop ratios '{ratios}', expected (0, 1]")
            return None
        return (ratio_w, ratio_h)

    @staticmethod
    def _center_crop_stereo_pair(
        imgs_left: np.ndarray,
        imgs_right: np.ndarray,
        crop_ratio,
        K: np.ndarray,
    ):
        ratio_w, ratio_h = crop_ratio
        if imgs_left.shape[1:3] != imgs_right.shape[1:3]:
            print("[policy] Stereo crop skipped: left/right image sizes do not match")
            return imgs_left, imgs_right, K
        height, width = imgs_left.shape[1:3]
        crop_w = max(1, int(round(width * ratio_w)))
        crop_h = max(1, int(round(height * ratio_h)))
        crop_w = min(crop_w, width)
        crop_h = min(crop_h, height)
        x0 = int((width - crop_w) / 2)
        y0 = int((height - crop_h) / 2)
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        imgs_left = imgs_left[:, y0:y1, x0:x1, :]
        imgs_right = imgs_right[:, y0:y1, x0:x1, :]
        K_crop = K.copy()
        K_crop[0, 2] -= x0
        K_crop[1, 2] -= y0
        return imgs_left, imgs_right, K_crop

    @staticmethod
    def _center_crop_rgb(
        imgs: np.ndarray,
        crop_ratio,
    ):
        ratio_w, ratio_h = crop_ratio
        height, width = imgs.shape[1:3]
        crop_w = max(1, int(round(width * ratio_w)))
        crop_h = max(1, int(round(height * ratio_h)))
        crop_w = min(crop_w, width)
        crop_h = min(crop_h, height)
        x0 = int((width - crop_w) / 2)
        y0 = int((height - crop_h) / 2)
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        return imgs[:, y0:y1, x0:x1, :]

    def get_real_umi_action(
            self,
            action: np.ndarray,
            env_obs: Dict[str, np.ndarray],
        ):
        env_action: Dict[str, np.ndarray] = {}
        if self.reference_frame == 'world':
            for robot_idx in range(len(self.id_to_side)):
                prefix = self.id_to_side[robot_idx]
                tf_key = prefix + '_tf'
                if tf_key not in env_obs:
                    continue
                pose_mat = np.asarray(env_obs[tf_key], dtype=float)[-1]

                action_pose10d = action[..., self.action_indexing[f'{prefix}_eef_pos'][0]: self.action_indexing[f'{prefix}_eef_rot_axis_angle'][1]]
                action_pose_mat = pose10d_to_mat(action_pose10d)

                action_mat = convert_pose_mat_rep(
                    action_pose_mat,
                    base_pose_mat=pose_mat,
                    pose_rep=self.action_pose_repr,
                    backward=True)

                env_action[f'{prefix}_tf'] = action_mat
                if prefix + '_gripper_width' in self.action_indexing.keys():
                    action_grip = action[..., self.action_indexing[f'{prefix}_gripper_width'][0]: self.action_indexing[f'{prefix}_gripper_width'][1]]
                    env_action[f'{prefix}_gripper_width'] = action_grip
        else:
            print(f'using reference frame: {self.reference_frame}')
            assert self.reference_frame == 'left' or self.reference_frame == 'right' or self.reference_frame == 'head', f'unsupported reference frame {self.reference_frame}'
            base_prefix = f'gripper_{self.reference_frame}'
            base_tf_key = base_prefix + '_tf'
            if base_tf_key not in env_obs:
                print(f"[policy] Missing base transform '{base_tf_key}' in observation; cannot transform actions.")
                return env_action
            base_pose_mat = np.asarray(env_obs[base_tf_key], dtype=float)[-1]

            for robot_idx in range(len(self.id_to_side)):
                prefix = self.id_to_side[robot_idx]
                if f'{prefix}_eef_pos' not in self.action_indexing or f'{prefix}_eef_rot_axis_angle' not in self.action_indexing:
                    continue
                action_pose10d = action[..., self.action_indexing[f'{prefix}_eef_pos'][0]: self.action_indexing[f'{prefix}_eef_rot_axis_angle'][1]]
                action_pose_mat = pose10d_to_mat(action_pose10d)
                action_mat = convert_pose_mat_rep(
                    action_pose_mat,
                    base_pose_mat=base_pose_mat,
                    pose_rep=self.action_pose_repr,
                    backward=True)
                env_action[f'{prefix}_tf'] = action_mat
                if prefix + '_gripper_width' in self.action_indexing.keys():
                    action_grip = action[..., self.action_indexing[f'{prefix}_gripper_width'][0]: self.action_indexing[f'{prefix}_gripper_width'][1]]
                    env_action[f'{prefix}_gripper_width'] = action_grip
                    # print(f"[policy] action {prefix}_gripper_width: {action_grip}")

            if 'camera_head_lookatpoint' in self.action_indexing:
                lookatpoint_action = action[..., self.action_indexing['camera_head_lookatpoint'][0]: self.action_indexing['camera_head_lookatpoint'][1]]
                lookatpoint_action_homogeneous = np.concatenate([
                    lookatpoint_action,
                    np.ones((lookatpoint_action.shape[0], 1), dtype=lookatpoint_action.dtype)
                ], axis=-1)  # (H, 4)
                # transform from base frame to world frame
                lookatpoint_world = base_pose_mat @ lookatpoint_action_homogeneous.T  # (4, H)
                lookatpoint_world = lookatpoint_world.T  # (H, 4)
                env_action['camera_head_lookatpoint'] = lookatpoint_world[:, :3]

        return env_action

    def run_node(self, plot_actions: bool = False, send_raw_actions: bool = False):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode is listening on {self.ip}:{self.port}")
        while True:
            try:
                raw_msg = socket.recv()
            except Exception:
                err = echo_exception()
                print(f"[policy] Failed to receive message: {err}")
                socket.send_pyobj({'error': f'recv_failed: {err}'})
                continue

            # Handle string control messages (e.g., get_obs_keys) in the same REQ/REP cycle.
            try:
                obs = dill.loads(raw_msg)
            except Exception:
                try:
                    msg_str = raw_msg.decode('utf-8')
                except Exception:
                    err = echo_exception()
                    print(f"[policy] Unable to decode incoming message: {err}")
                    socket.send_pyobj({'error': f'decode_failed: {err}'})
                    continue

                print(f"[policy] Received control message: {msg_str}")
                if msg_str == "get_obs_keys":
                    socket.send_pyobj(self.required_obs_keys)
                    print("[policy] Sent required observation keys.")
                else:
                    socket.send_pyobj({'error': f'unhandled_message: {msg_str}'})
                continue

            obs_timestamps = obs['timestamp']
            action_tf = None
            try:
                infer_start = time.monotonic()
                obs_dict_np = self.get_real_umi_obs_dict(
                    env_obs=obs.copy(),
                    tx_robot1_robot0=None,  # this should be passed in None since the poses are already in the same frame (are from the same robot)
                    episode_start_pose=None,  # this should be passed in None since we don't need this
                )
                raw_action = self.predict_action(obs_dict_np)
                action_tf = self.get_real_umi_action(
                    action=raw_action,
                    env_obs=obs,    # note: should pass in obs rather than obs_dict_np, to transform to world frame
                )
                if self._debug_attn_queue is not None:
                    vit_viz = {}
                    pointmap_viz = {}
                    obs_encoder = getattr(self.policy, "obs_encoder", None)
                    if obs_encoder is not None:
                        vit_viz = getattr(obs_encoder, "_last_vit_viz", {}) or {}
                        pointmap_viz = getattr(obs_encoder, "_last_3d_viz", {}) or {}
                    gripper_poses = self._get_debug_gripper_poses(obs)
                    lookat_point = self._get_debug_lookat_point(raw_action, obs)
                    _enqueue_debug_attn(
                        debug_queue=self._debug_attn_queue,
                        debug_counter=self._debug_attn_counter,
                        output_root=self._debug_attn_dir,
                        pointmaps=self._last_pointmaps,
                        pointmap_masks=self._last_pointmap_masks,
                        pointmaps_ref=self._last_pointmaps_ref,
                        pointmap_rgbs=self._last_pointmap_rgbs,
                        gripper_poses=gripper_poses,
                        lookat_point=lookat_point,
                        vit_viz=vit_viz,
                        pointmap_viz=pointmap_viz,
                    )
                if not action_tf:
                    print("[policy] Empty action dict; skipping reply")
                    continue
                infer_latency = time.monotonic() - infer_start
                print(f'Inference time: {infer_latency:.3f} s')
                # # print gripper width for debugging
                # for key in action_tf.keys():
                #     if 'gripper_width' in key:
                #         print(f'[policy] action {key}: {action_tf[key]}')
            except Exception as e:
                err_str = echo_exception()
                print(f'Error: {err_str}')
                socket.send_pyobj({'error': err_str})
                continue
            if not action_tf:
                socket.send_pyobj({'error': 'empty_action'})
                continue
            first_key = next(iter(action_tf.keys()))
            horizon = np.asarray(action_tf[first_key]).shape[0]
            action_timestamps = (np.arange(horizon, dtype=np.float64)
                                 ) * self.dt + obs_timestamps[-1]
            if send_raw_actions:
                raw_actions_tf = {k: np.asarray(v) for k, v in action_tf.items()}
                raw_action_timestamps = action_timestamps.copy()
            # action_exec_latency = 0.0
            # curr_time = time.monotonic()
            # is_new = action_timestamps > (curr_time + action_exec_latency)
            # if np.sum(is_new) == 0:
            #     # exceeded time budget, still do something
            #     action_tf = {k: np.asarray(v)[[-1]] for k, v in action_tf.items()}
            #     # schedule on next available step
            #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / self.dt))
            #     action_timestamp = eval_t_start + (next_step_idx) * self.dt
            #     print('Over budget', action_timestamp - curr_time)
            #     action_timestamps = np.array([action_timestamp])
            # else:
            #     action_tf = {k: np.asarray(v)[is_new] for k, v in action_tf.items()}
            #     action_timestamps = action_timestamps[is_new]
            # socket.send_pyobj(action)
            # send both action and timestamps
            payload = {
                'actions_tf': action_tf,
                'timestamps': action_timestamps,
            }
            if send_raw_actions:
                payload['raw_actions_tf'] = raw_actions_tf
                payload['raw_timestamps'] = raw_action_timestamps
                # print(f"[policy] Sending raw actions with timestamps: {raw_action_timestamps}")
            socket.send_pyobj(payload)

def run_policy_inference_node(input, ip, port, device, frequency, action_horizon, pointmap_scale, intrinsic_file, fs_ckpt_dir, vis_attn=False, debug_save_attn=False, plot_actions=False, send_raw_actions=False, reference_frame=None, head_stereo_crop=None, stereo_rectify=False, fisheye_calibration_json=None, stereo_calibration_json=None, save_rectified=False, debug_stereo_k=False):
    node = PolicyInferenceNode(
        input,
        ip,
        port,
        device,
        frequency,
        action_horizon,
        pointmap_scale=pointmap_scale,
        intrinsic_file=intrinsic_file,
        fs_ckpt_dir=fs_ckpt_dir,
        vis_attn=vis_attn,
        debug_save_attn=debug_save_attn,
        reference_frame=reference_frame,
        head_stereo_crop=head_stereo_crop,
        stereo_rectify=stereo_rectify,
        fisheye_calibration_json=fisheye_calibration_json,
        stereo_calibration_json=stereo_calibration_json,
        save_rectified=save_rectified,
        debug_stereo_k=debug_stereo_k)
    node.run_node(plot_actions=plot_actions, send_raw_actions=send_raw_actions)

def run_policy_on_dataset_sample(ckpt_path: str, ip, port, device: str, frequency: float, use_gt_action: bool = False, action_horizon: int = 8, intrinsic_file: str = '../rainbow/K_head.txt', fs_ckpt_dir: str = '.../deps/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth', vis_attn: bool = False, debug_save_attn: bool = False, head_stereo_crop=None, stereo_rectify=False, fisheye_calibration_json=None, stereo_calibration_json=None, save_rectified=False, debug_stereo_k=False):
    print(f"Running offline sample inference using checkpoint: {ckpt_path}")
    node = PolicyInferenceNode(ckpt_path, ip=ip, port=port, device=device, frequency=frequency, action_horizon=action_horizon, use_gt_action=use_gt_action, intrinsic_file=intrinsic_file, fs_ckpt_dir=fs_ckpt_dir, vis_attn=vis_attn, debug_save_attn=debug_save_attn, head_stereo_crop=head_stereo_crop, stereo_rectify=stereo_rectify, fisheye_calibration_json=fisheye_calibration_json, stereo_calibration_json=stereo_calibration_json, save_rectified=save_rectified, debug_stereo_k=debug_stereo_k)

    # Load dataset from the checkpoint config
    dataset_cfg = node.cfg.task.dataset
    dataset_cfg.dataset_path = '/home/xiaomeng/hommi/hommi/demonstration_processing/tmp_sessions/cup-realab-debug/replay_buffer_cup-realab-debug.zarr.zip'
    dataset = hydra.utils.instantiate(dataset_cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{node.ip}:{node.port}")
    print(f"PolicyInferenceNode is listening on {node.ip}:{node.port}")
    # send obs keys as well
    while True:
        msg = socket.recv_string()
        print(f"[policy] Received message: {msg}")
        if msg == "get_obs_keys":
            break
        else:
            print("[policy] Waiting for 'get_obs_keys' message")
    socket.send_pyobj(node.required_obs_keys)
    print(f"[policy] Sent required observation keys.")

    for sample_idx, batch in enumerate(dataloader):
        if sample_idx % 24 != 0 or sample_idx > 400:
            continue
        obs = socket.recv_pyobj()
        obs_timestamps = obs['timestamp']
        if isinstance(obs, dict):
            for timestamp in obs_timestamps:
                print(f"[policy] received obs of {timestamp:.3f}")
        else:
            continue
        print(f"[policy] received obs {obs}")
        if use_gt_action:
            # Use ground truth action from the dataset
            raw_action = batch['action'][0].detach().to('cpu').numpy()[:action_horizon, ...]
            print(f"[policy] using ground truth action from dataset: {raw_action.shape}{raw_action}")
        else:
            # Predict action using the policy
            raw_action = node.policy.predict_action(batch['obs'])['action_pred'][0].detach().to('cpu').numpy()[:action_horizon, ...]
            print(f"[policy] using ground truth obs {batch['obs']} to predict action")

        action_tf = node.get_real_umi_action(
            action=raw_action,
            env_obs=obs,
        )
        # print sent action
        print("[policy] sending action tf dict")
        action_len = np.asarray(next(iter(action_tf.values()))).shape[0]
        action_timestamps = (np.arange(action_len, dtype=np.float64)) * node.dt + obs_timestamps[-1]
        socket.send_pyobj({
            'actions_tf': action_tf,
            'timestamps': action_timestamps,
        })

@click.command()
@click.option('--policy_input', '-i', required=True, help='Path to policy checkpoint', default='/home/xiaomengxu/hommi/hommi/train_network/runs/2025.07.30/01.39.19_umi_policy_bimanual_umi_policy_diffusion_transformer_timm_default')
@click.option('--ip', default="0.0.0.0")
@click.option('--policy_port', default=8766, help="Port to listen on for policy inference")
@click.option('--device', default="cuda", help="Device to run on")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--offline_sample', is_flag=True, help="Run a single inference on a sample from the dataset instead of launching server")
@click.option('--use_gt_action', is_flag=True, help="Use ground truth action from the dataset instead of predicting action")
@click.option('--action_horizon', default=12, help="action horizon during inference.")
@click.option('--pointmap_scale', default=1.0, type=float, help="Scale factor for pointmap during inference (1.0 means no scaling)")
@click.option('--vis_attn', is_flag=True, help="Visualize attention maps during inference")
@click.option('--debug_save_attn', is_flag=True, help="Save pointmap/attention visualizations during inference")
@click.option('--intrinsic_file', default='./rainbow/K_head.txt', help="Path to intrinsic matrix file")
@click.option('--fisheye_calibration_json', default=None, type=str, help="Path to single-camera fisheye calibration json (K/D/DIM).")
@click.option('--stereo_calibration_json', default='/home/xiaomeng/rby1/camera/calibrate/camera_calibration_camera_head_main_rgb_camera_head_main_right_rgb/stereo_fisheye_calibration.json', type=str, help="Path to stereo fisheye calibration json.")
@click.option('--fs_ckpt_dir', default='/home/xiaomeng/hommi/deps/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth', type=str, help="Path to FoundationStereo checkpoint directory, if not provided, will use the default model")
@click.option('--send_raw_actions', is_flag=True, help="Include raw (unfiltered) policy actions and timestamps in the reply for plotting.")
@click.option('--reference_frame', default=None, type=str, help="overwrite reference frame in the checkpoint config, options: world / left / right / head")
@click.option('--head_stereo_crop', default=None, type=str, help="Center crop head stereo before pointmap as ratio or ratio_w,ratio_h (e.g., 0.8 or 0.8,0.9)")
@click.option('--stereo_rectify/--no-stereo_rectify', default=False, help="Enable rectification before stereo depth (uses calibration if available).")
@click.option('--save_rectified/--no-save_rectified', default=False, help="Save rectified stereo images for debugging.")
@click.option('--debug_stereo_k', is_flag=True, help="Print the K/baseline used for stereo depth.")
def main(policy_input, ip, policy_port, device, frequency, offline_sample, use_gt_action, action_horizon, pointmap_scale, intrinsic_file, fisheye_calibration_json, stereo_calibration_json, fs_ckpt_dir, vis_attn=False, debug_save_attn=False, send_raw_actions=False, reference_frame=None, head_stereo_crop=None, stereo_rectify=False, save_rectified=False, debug_stereo_k=False):
    head_stereo_crop = PolicyInferenceNode._normalize_crop_ratio(head_stereo_crop)
    if offline_sample:
        run_policy_on_dataset_sample(policy_input, ip, policy_port, device, frequency, use_gt_action=use_gt_action, action_horizon=action_horizon, intrinsic_file=intrinsic_file, fs_ckpt_dir=fs_ckpt_dir, vis_attn=vis_attn, debug_save_attn=debug_save_attn, head_stereo_crop=head_stereo_crop, stereo_rectify=stereo_rectify, fisheye_calibration_json=fisheye_calibration_json, stereo_calibration_json=stereo_calibration_json, save_rectified=save_rectified, debug_stereo_k=debug_stereo_k)
    else:
        run_policy_inference_node(
            policy_input,
            ip,
            policy_port,
            device,
            frequency,
            action_horizon=action_horizon,
            pointmap_scale=pointmap_scale,
            intrinsic_file=intrinsic_file,
            fisheye_calibration_json=fisheye_calibration_json,
            stereo_calibration_json=stereo_calibration_json,
            fs_ckpt_dir=fs_ckpt_dir,
            vis_attn=vis_attn,
            debug_save_attn=debug_save_attn,
            send_raw_actions=send_raw_actions,
            reference_frame=reference_frame,
            head_stereo_crop=head_stereo_crop,
            stereo_rectify=stereo_rectify,
            save_rectified=save_rectified,
            debug_stereo_k=debug_stereo_k,
            )

if __name__ == '__main__':
    main()
