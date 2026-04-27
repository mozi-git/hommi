import copy
import os
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Sequence, Union

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from hommi.common.cv_util import depth2xyzmap
from hommi.common.transform_util import pos_quat_xyzw_to_4x4
from umi.common.pose_util import pose_to_mat
from hommi.common.timecode_util import datetime_fromisoformat
# For writing PLY pointmaps we need the io module
from hommi.demonstration_processing.utils.depth_util import load_depth
from hommi.demonstration_processing.utils.generic_util import (
    demonstration_to_display_string,
    get_demonstration_json_data,
    read_demonstration_metadata,
)
from hommi.demonstration_processing.utils.gripper_util import iphone_to_tcp_poses
from hommi.demonstration_processing.utils.lookat_util import (
    compute_center_lookat_from_pointmap,
)

try:
    import open3d as o3d
    from open3d.visualization import rendering as o3d_rendering
except ImportError as exc:  # pragma: no cover - optional dependency
    o3d = None
    o3d_rendering = None
    OPEN3D_IMPORT_ERROR = exc
else:
    OPEN3D_IMPORT_ERROR = None


@dataclass
class PointmapVizOptions:
    iphone_calibration_path: str
    depth_shape: Tuple[int, int] = (192, 256)
    depth_side: str = "head"
    gripper_sides: Tuple[str, ...] = ("left", "right")
    frame_stride: int = 1
    max_frames: int = -1
    fps: float = 30.0
    min_depth: Optional[float] = 0.02
    max_depth: Optional[float] = 1.2
    max_points: int = 48000
    point_size: float = 3.0
    axis_length: float = 0.08
    renderer_width: int = 1280
    renderer_height: int = 720
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.6)
    eye: Tuple[float, float, float] = (0.0, 0.5, -0.2)
    up: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    camera_distance_scale: float = 1.5
    min_camera_radius: float = 0.12
    rotate_head_poses: bool = False
    lookat_marker_radius: float = 0.02
    use_rgb_colors: bool = True
    merge_wrist_pointclouds: bool = False
    wrist_sides: Tuple[str, ...] = ("left", "right")
    save_combined_pointmap: bool = False
    combined_pointmap_pattern: str = "{depth_side}_combined_pointmap_{frame:05d}.ply"
    skip_if_exists: bool = True
    overwrite: bool = False
    verbose: bool = False
    reference_frame: str = "left"
    reference_side: str = "left"
    video_fourcc: str = "avc1"
    use_fixed_camera_pose: bool = True


@dataclass
class PointmapSequence:
    depth_frames: Optional[np.ndarray]
    head_pointmaps: Optional[np.ndarray]
    head_poses: np.ndarray
    head_is_lost: np.ndarray
    rgb_frames: List[np.ndarray]
    pose_to_rgb_idx: Optional[np.ndarray]
    gripper_data: Dict[str, Dict[str, np.ndarray]]
    wrist_resources: Dict[str, Dict]
    lookat_points: Optional[np.ndarray] = None


class EpisodeArrayView:
    """Lightweight view over a 1D zarr array limited to a single episode."""

    def __init__(self, array, episode_slice: slice):
        self._array = array
        self._start = 0 if episode_slice.start is None else episode_slice.start
        self._stop = episode_slice.stop

    def __len__(self) -> int:
        return max(0, self._stop - self._start)

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        length = len(self)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(length)
            actual = slice(self._start + start, self._start + stop, step)
            return np.asarray(self._array[actual])

        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(idx)
        return np.asarray(self._array[self._start + idx])


class _RGBFrameAccessor:
    """Adapter that exposes RGB frames in BGR order expected by OpenCV routines."""

    def __init__(self, view: EpisodeArrayView, stored_format: str = "bgr"):
        self._view = view
        self._stored_format = stored_format.lower()

    def __len__(self) -> int:
        return len(self._view)

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        frame = np.asarray(self._view[idx])
        if self._stored_format == "rgb":
            frame = frame[..., ::-1]
        return frame


_ARKIT_TO_CAMERA = np.eye(4, dtype=np.float32)
_ARKIT_TO_CAMERA[1, 1] = -1.0
_ARKIT_TO_CAMERA[2, 2] = -1.0


def _load_intrinsic_matrix(calibration_path: str) -> np.ndarray:
    if not os.path.isabs(calibration_path):
        calibration_path = os.path.abspath(calibration_path)
    with open(calibration_path, "r") as f:
        intrinsic = np.array(
            yaml.safe_load(f)["depth"]["intrinsicMatrix"], dtype=np.float32
        )
    return intrinsic


def _load_demo_sequence(
    demonstration_dir: str, options: PointmapVizOptions
) -> Tuple[PointmapSequence, np.ndarray]:
    depth_path = os.path.join(demonstration_dir, f"{options.depth_side}_depth.raw")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth data missing at {depth_path}")

    head_csv = os.path.join(
        demonstration_dir, f"camera_trajectory_iphone_{options.depth_side}.csv"
    )
    if not os.path.exists(head_csv):
        raise FileNotFoundError(f"Head trajectory missing at {head_csv}")

    intrinsic = _load_intrinsic_matrix(options.iphone_calibration_path)

    depth_frames = load_depth(
        depth_path,
        depth_shape=options.depth_shape,
        dtype=np.float16,
    ).astype(np.float32)

    head_df = pd.read_csv(head_csv)
    head_is_lost = head_df["is_lost"].values.astype(bool)
    head_poses = pos_quat_xyzw_to_4x4(
        head_df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values.astype(np.float32)
    )
    head_poses = iphone_to_tcp_poses(
        head_poses, not_use_iphone_tcp_offset=True, rotate_z_pi=False
    )

    gripper_data: Dict[str, Dict[str, np.ndarray]] = {}
    for side in options.gripper_sides:
        csv_path = os.path.join(
            demonstration_dir, f"camera_trajectory_iphone_{side}.csv"
        )
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        poses = pos_quat_xyzw_to_4x4(
            df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values.astype(np.float32)
        )
        tcp_poses = iphone_to_tcp_poses(
            poses, not_use_iphone_tcp_offset=True, rotate_z_pi=False
        )
        gripper_data[side] = {
            "poses": tcp_poses,
            "is_lost": df["is_lost"].values.astype(bool),
        }

    if not gripper_data:
        raise RuntimeError(
            f"No gripper trajectories found in {demonstration_dir}. "
            "Expected camera_trajectory_iphone_left/right.csv."
        )

    pose_to_rgb_idx = None
    rgb_frames: List[np.ndarray] = []
    if options.use_rgb_colors:
        rgb_path = os.path.join(demonstration_dir, f"{options.depth_side}_rgb.mp4")
        if os.path.exists(rgb_path):
            rgb_frames = _load_video_frames(rgb_path)
            try:
                pose_to_rgb_idx = _build_pose_to_rgb_indices(
                    demonstration_dir, options.depth_side, len(head_poses)
                )
            except Exception as exc:
                print(f"[pointmap_viz] Warning: failed to build RGB mapping ({exc})")
                pose_to_rgb_idx = None

    wrist_resources = _load_wrist_resources(
        demonstration_dir, options, head_poses.shape[0]
    )

    sequence = PointmapSequence(
        depth_frames=depth_frames,
        head_poses=head_poses,
        head_is_lost=head_is_lost,
        rgb_frames=rgb_frames,
        pose_to_rgb_idx=pose_to_rgb_idx,
        gripper_data=gripper_data,
        wrist_resources=wrist_resources,
    )
    return sequence, intrinsic

def _set_camera_projection(camera: o3d_rendering.Camera, width: int, height: int):
    fov = 60.0
    aspect_ratio = width / height
    near_plane, far_plane = 0.01, 5.0
    try:
        fov_type = getattr(o3d_rendering.Camera.FovType, "Vertical", None)
        if fov_type is not None:
            camera.set_projection(fov, aspect_ratio, near_plane, far_plane, fov_type)
        else:
            raise AttributeError
    except (AttributeError, TypeError):
        camera.set_projection(fov, aspect_ratio, near_plane, far_plane)


def _align_z_to(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.eye(3)
    direction /= norm
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(direction, z_axis):
        return np.eye(3)
    if np.allclose(direction, -z_axis):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return rot


def _create_axis_templates(axis_length: float):
    """Pre-build arrow meshes for RGB axes."""
    specs = (
        ("x", np.array([1.0, 0.0, 0.0]), (1.0, 0.1, 0.1)),
        ("y", np.array([0.0, 1.0, 0.0]), (0.2, 0.8, 0.2)),
        ("z", np.array([0.0, 0.0, 1.0]), (0.2, 0.3, 1.0)),
    )
    templates = {}
    cyl_radius = axis_length * 0.03
    cone_radius = axis_length * 0.06
    cylinder_height = axis_length * 0.65
    cone_height = axis_length * 0.35

    for axis_name, direction, color in specs:
        mesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cyl_radius,
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
        )
        mesh.paint_uniform_color(color)
        transform = np.eye(4)
        transform[:3, :3] = _align_z_to(direction)
        mesh.transform(transform)
        mesh.compute_vertex_normals()
        templates[axis_name] = mesh
    return templates


def _clone_geometry(geom):
    if hasattr(geom, "clone"):
        return geom.clone()
    return copy.deepcopy(geom)


def _compute_camera_pose(points: np.ndarray, options: PointmapVizOptions):
    if options.use_fixed_camera_pose:
        lookat = np.array(options.lookat, dtype=np.float64)
        eye = np.array(options.eye, dtype=np.float64)
        if np.allclose(lookat, eye):
            raise ValueError("lookat and eye must differ when use_fixed_camera_pose=True")
        return lookat, eye

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.max(bbox_max - bbox_min)
    radius = max(extent * options.camera_distance_scale, options.min_camera_radius)
    eye_offset = np.array([0.0, -radius, radius * 0.35])
    eye = center + eye_offset
    return center, eye


def _mesh_vertices_with_color(mesh: o3d.geometry.TriangleMesh):
    verts = np.asarray(mesh.vertices)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
    else:
        colors = np.ones_like(verts) * np.array([0.8, 0.8, 0.8])
    return verts.copy(), colors.copy()


def _load_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _build_pose_to_rgb_indices(
        demonstration_dir: str,
        depth_side: str,
        n_pose_frames: int,
    ) -> Optional[np.ndarray]:
    metadata = read_demonstration_metadata(demonstration_dir)
    pose_time_strs = metadata.get(f"{depth_side}_frame_times")
    if not pose_time_strs:
        return None
    pose_ts = np.array(
        [datetime_fromisoformat(t).timestamp() for t in pose_time_strs[:n_pose_frames]],
        dtype=np.float64,
    )
    rgb_json = get_demonstration_json_data(demonstration_dir, depth_side)
    rgb_times = rgb_json.get("rgbTimes")
    if not rgb_times:
        return None
    rgb_ts = np.array([datetime_fromisoformat(t).timestamp() for t in rgb_times], dtype=np.float64)
    pose_to_rgb_idx = np.searchsorted(rgb_ts, pose_ts, side="right") - 1
    pose_to_rgb_idx = np.clip(pose_to_rgb_idx, 0, len(rgb_ts) - 1)
    return pose_to_rgb_idx


def _build_pose_to_side_indices(
    head_ts: np.ndarray,
    side_ts: np.ndarray,
    max_idx: int,
) -> np.ndarray:
    indices = np.searchsorted(side_ts, head_ts, side="right") - 1
    indices = np.clip(indices, 0, max_idx - 1)
    return indices


def _load_wrist_resources(
    demonstration_dir: str,
    options: PointmapVizOptions,
    head_pose_count: int,
) -> Dict[str, Dict]:
    resources = {}
    if not options.merge_wrist_pointclouds:
        return resources

    metadata = read_demonstration_metadata(demonstration_dir)
    head_time_strs = metadata.get(f"{options.depth_side}_frame_times")
    if not head_time_strs:
        print("[pointmap_viz] No head frame times found; skipping wrist pointcloud merge.")
        return resources
    head_ts = np.array(
        [datetime_fromisoformat(t).timestamp() for t in head_time_strs[:head_pose_count]],
        dtype=np.float64,
    )

    wrist_colors = {
        "left": np.array([0.15, 0.45, 0.95]),
        "right": np.array([0.95, 0.35, 0.15]),
    }

    for side in options.wrist_sides:
        depth_path = os.path.join(demonstration_dir, f"{side}_depth.raw")
        pose_csv = os.path.join(demonstration_dir, f"camera_trajectory_iphone_{side}.csv")
        if not (os.path.exists(depth_path) and os.path.exists(pose_csv)):
            continue
        side_time_strs = metadata.get(f"{side}_frame_times")
        if not side_time_strs:
            continue
        try:
            depth_array = load_depth(
                depth_path,
                depth_shape=options.depth_shape,
                dtype=np.float16,
            )
        except Exception as exc:
            print(f"[pointmap_viz] Failed to load depth for {side}: {exc}")
            continue

        side_df = pd.read_csv(pose_csv)
        poses = pos_quat_xyzw_to_4x4(
            side_df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values.astype(np.float32)
        )
        poses = iphone_to_tcp_poses(poses, not_use_iphone_tcp_offset=True, rotate_z_pi=False)
        n_frames = min(len(side_time_strs), len(depth_array), len(poses))
        if n_frames == 0:
            continue
        side_ts = np.array(
            [datetime_fromisoformat(t).timestamp() for t in side_time_strs[:n_frames]],
            dtype=np.float64,
        )
        mapping = _build_pose_to_side_indices(head_ts, side_ts, n_frames)
        rgb_path = os.path.join(demonstration_dir, f"{side}_rgb.mp4")
        rgb_frames = _load_video_frames(rgb_path) if os.path.exists(rgb_path) else []
        depth_to_rgb_idx = None
        if rgb_frames:
            depth_to_rgb_idx = _build_pose_to_rgb_indices(
                demonstration_dir, side, n_frames
            )
        resources[side] = {
            "depth": depth_array[:n_frames],
            "poses": poses[:n_frames],
            "mapping": mapping,
            "color": wrist_colors.get(side, np.array([0.6, 0.6, 0.2])),
            "rgb_frames": rgb_frames,
            "depth_to_rgb": depth_to_rgb_idx,
        }
    return resources


def _load_demo_sequence(
    demonstration_dir: str, options: PointmapVizOptions
) -> Tuple[PointmapSequence, np.ndarray]:
    depth_path = os.path.join(demonstration_dir, f"{options.depth_side}_depth.raw")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth data missing at {depth_path}")

    head_csv = os.path.join(
        demonstration_dir, f"camera_trajectory_iphone_{options.depth_side}.csv"
    )
    if not os.path.exists(head_csv):
        raise FileNotFoundError(f"Head trajectory missing at {head_csv}")

    intrinsic = _load_intrinsic_matrix(options.iphone_calibration_path)

    depth_frames = load_depth(
        depth_path,
        depth_shape=options.depth_shape,
        dtype=np.float16,
    ).astype(np.float32)

    head_df = pd.read_csv(head_csv)
    head_is_lost = head_df["is_lost"].values.astype(bool)
    head_poses = pos_quat_xyzw_to_4x4(
        head_df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values.astype(np.float32)
    )
    head_poses = iphone_to_tcp_poses(
        head_poses, not_use_iphone_tcp_offset=True, rotate_z_pi=False
    )

    gripper_data: Dict[str, Dict[str, np.ndarray]] = {}
    for side in options.gripper_sides:
        csv_path = os.path.join(
            demonstration_dir, f"camera_trajectory_iphone_{side}.csv"
        )
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        poses = pos_quat_xyzw_to_4x4(
            df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values.astype(np.float32)
        )
        tcp_poses = iphone_to_tcp_poses(
            poses, not_use_iphone_tcp_offset=True, rotate_z_pi=False
        )
        gripper_data[side] = {
            "poses": tcp_poses,
            "is_lost": df["is_lost"].values.astype(bool),
        }

    if not gripper_data:
        raise RuntimeError(
            f"No gripper trajectories found in {demonstration_dir}. "
            "Expected camera_trajectory_iphone_left/right.csv."
        )

    rgb_frames: List[np.ndarray] = []
    pose_to_rgb_idx = None
    if options.use_rgb_colors:
        rgb_path = os.path.join(demonstration_dir, f"{options.depth_side}_rgb.mp4")
        if os.path.exists(rgb_path):
            rgb_frames = _load_video_frames(rgb_path)
            try:
                pose_to_rgb_idx = _build_pose_to_rgb_indices(
                    demonstration_dir, options.depth_side, len(head_poses)
                )
            except Exception as exc:
                print(f"[pointmap_viz] Warning: failed to build RGB mapping ({exc})")
                pose_to_rgb_idx = None

    wrist_resources = _load_wrist_resources(
        demonstration_dir, options, head_poses.shape[0]
    )

    sequence = PointmapSequence(
        depth_frames=depth_frames,
        head_pointmaps=None,
        head_poses=head_poses,
        head_is_lost=head_is_lost,
        rgb_frames=rgb_frames,
        pose_to_rgb_idx=pose_to_rgb_idx,
        gripper_data=gripper_data,
        wrist_resources=wrist_resources,
    )
    return sequence, intrinsic


def render_pointmap_alignment_video(
    demonstration_dir: str,
    output_path: str,
    options: PointmapVizOptions,
) -> Optional[str]:
    """
    Render a video showing the head pointmap and gripper frames in the head frame.
    """

    if OPEN3D_IMPORT_ERROR is not None:
        raise RuntimeError(
            "open3d is required for pointmap visualization but is not installed"
        ) from OPEN3D_IMPORT_ERROR

    display_name = demonstration_to_display_string(
        demonstration_dir, options.depth_side
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and options.skip_if_exists and not options.overwrite:
        if options.verbose:
            print(f"[pointmap_viz] Skipping existing video at {output_path}")
        return None

    sequence, intrinsic = _load_demo_sequence(demonstration_dir, options)

    total_frames = len(sequence.head_poses)
    if options.max_frames > 0:
        total_frames = min(total_frames, options.max_frames)
    frame_indices = list(range(0, total_frames, options.frame_stride))

    if not frame_indices:
        raise RuntimeError("No frames available for visualization.")

    render_pointmap_sequence(
        display_name,
        sequence,
        intrinsic,
        output_path,
        frame_indices,
        options,
        save_root=demonstration_dir,
    )
    return output_path


def _get_reference_pose(
    sequence: PointmapSequence, options: PointmapVizOptions, idx: int, head_pose: np.ndarray
) -> Optional[np.ndarray]:
    frame_mode = options.reference_frame.lower()
    if frame_mode == "head":
        return head_pose

    if frame_mode == "gripper":
        side = options.reference_side
        data = sequence.gripper_data.get(side)
        if not data:
            print(f"[pointmap_viz] Missing gripper data for side '{side}'")
            return None
        pose_idx = min(idx, len(data["poses"]) - 1)
        if data["is_lost"][pose_idx]:
            return None
        return data["poses"][pose_idx]

    raise ValueError(f"Unsupported reference_frame '{options.reference_frame}'")


def _get_rgb_frame(rgb_frames, idx: int) -> Optional[np.ndarray]:
    if rgb_frames is None:
        return None
    if len(rgb_frames) == 0:
        return None
    if idx < 0 or idx >= len(rgb_frames):
        return None
    frame = rgb_frames[idx]
    return np.asarray(frame)


def _extract_head_pointcloud(
    sequence: PointmapSequence,
    idx: int,
    intrinsic: np.ndarray,
    options: PointmapVizOptions,
    display_name: str,
    need_lookat: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    xyz_map = None
    depth = None
    depth_source = None
    if sequence.head_pointmaps is not None:
        xyz_map = np.asarray(sequence.head_pointmaps[idx])
        depth = xyz_map[..., 2]
        depth_source = "pointmap"
    elif sequence.depth_frames is not None:
        depth = np.asarray(sequence.depth_frames[idx])
        xyz_map = depth2xyzmap(depth, intrinsic)
        depth_source = "depth"
    else:
        return None

    valid_mask = depth > 0
    if options.min_depth is not None:
        valid_mask &= depth >= options.min_depth
    if options.max_depth is not None and options.max_depth > 0:
        valid_mask &= depth <= options.max_depth

    head_points = xyz_map[valid_mask]
    if head_points.size == 0 and depth_source == "pointmap" and sequence.depth_frames is not None:
        depth = np.asarray(sequence.depth_frames[idx])
        xyz_map = depth2xyzmap(depth, intrinsic)
        valid_mask = depth > 0
        if options.min_depth is not None:
            valid_mask &= depth >= options.min_depth
        if options.max_depth is not None and options.max_depth > 0:
            valid_mask &= depth <= options.max_depth
        head_points = xyz_map[valid_mask]

    used_fallback = False
    if head_points.size == 0:
        valid_mask = depth > 0
        head_points = xyz_map[valid_mask]
        if head_points.size == 0:
            return None
        used_fallback = True
    flat_mask = valid_mask.reshape(-1)

    derived_lookat = None
    if need_lookat:
        derived_lookat = compute_center_lookat_from_pointmap(xyz_map).astype(np.float16)

    pose_to_rgb_idx = sequence.pose_to_rgb_idx
    rgb_frames = sequence.rgb_frames
    use_rgb_colors = (
        options.use_rgb_colors
        and pose_to_rgb_idx is not None
        and len(rgb_frames) > 0
        and idx < len(pose_to_rgb_idx)
    )

    colors = None
    if use_rgb_colors:
        rgb_index = int(pose_to_rgb_idx[idx])
        rgb_frame = _get_rgb_frame(rgb_frames, rgb_index)
        if rgb_frame is not None:
            rgb_resized = cv2.resize(
                rgb_frame,
                (options.depth_shape[1], options.depth_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
            flat_rgb = (rgb_resized.reshape(-1, 3).astype(np.float32) / 255.0)
            colors = flat_rgb[flat_mask]

    if colors is None:
        if used_fallback and options.verbose:
            print(
                f"[pointmap_viz] {display_name} frame {idx}: depth threshold relaxed"
            )
        z_vals = head_points[:, 2]
        z_min = z_vals.min() if options.min_depth is None else options.min_depth
        if options.max_depth is None or options.max_depth <= 0 or used_fallback:
            z_max = z_vals.max()
        else:
            z_max = options.max_depth
        z_norm = (z_vals - z_min) / max(z_max - z_min, 1e-4)
        colors = np.stack(
            [
                z_norm,
                1.0 - z_norm,
                np.clip(1.0 - np.abs(z_norm - 0.5) * 2.0, 0.0, 1.0),
            ],
            axis=1,
        )

    return head_points, colors.astype(np.float64, copy=False), derived_lookat


def render_pointmap_sequence(
    display_name: str,
    sequence: PointmapSequence,
    intrinsic: np.ndarray,
    output_path: str,
    frame_indices: Sequence[int],
    options: PointmapVizOptions,
    save_root: Optional[str] = None,
) -> str:
    """Shared renderer used by both raw demo and replay-buffer visualizations."""
    renderer = o3d_rendering.OffscreenRenderer(options.renderer_width, options.renderer_height)
    renderer.scene.set_background(list(options.background_color))
    _set_camera_projection(renderer.scene.camera, options.renderer_width, options.renderer_height)

    point_material = o3d_rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = options.point_size

    frame_material = o3d_rendering.MaterialRecord()
    frame_material.shader = "defaultUnlit"

    axis_templates = _create_axis_templates(options.axis_length)
    rng = np.random.default_rng(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = options.video_fourcc
    if not isinstance(fourcc, str) or len(fourcc) != 4:
        raise ValueError(f"video_fourcc must be a 4-character string, got {fourcc}")
    video_out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*fourcc),
        options.fps,
        (options.renderer_width, options.renderer_height),
    )
    frames_rendered = 0
    pointmap_written = False
    save_root = save_root or os.path.dirname(output_path)

    try:
        head_poses = sequence.head_poses
        head_is_lost = sequence.head_is_lost
        wrist_resources = sequence.wrist_resources or {}
        gripper_data = sequence.gripper_data or {}

        with tqdm(frame_indices, leave=False, desc=f"[pointmap_viz] {display_name}") as pbar:
            for idx in frame_indices:
                if idx >= len(head_poses):
                    break
                if head_is_lost[idx]:
                    if options.verbose:
                        print(f"[pointmap_viz] frame {idx} skipped: head pose marked lost")
                    pbar.update(1)
                    continue

                head_pose = head_poses[idx]
                reference_pose = _get_reference_pose(sequence, options, idx, head_pose)
                if reference_pose is None:
                    if options.verbose:
                        print(f"[pointmap_viz] frame {idx} skipped: reference pose unavailable")
                    pbar.update(1)
                    continue
                ref_T_world = np.linalg.inv(reference_pose)

                need_lookat = sequence.lookat_points is None
                head_points_colors = _extract_head_pointcloud(
                    sequence, idx, intrinsic, options, display_name, need_lookat=need_lookat
                )
                if head_points_colors is None:
                    if options.verbose:
                        print(f"[pointmap_viz] frame {idx} skipped: no valid head pointcloud")
                    pbar.update(1)
                    continue
                head_points, colors, derived_lookat = head_points_colors

                head_T_ref = ref_T_world @ head_pose
                points_ref = (
                    head_T_ref[:3, :3] @ head_points.T + head_T_ref[:3, 3:4]
                ).T

                points_list = [points_ref]
                colors_list = [colors]

                lookat_point_ref = None
                if sequence.lookat_points is not None and idx < len(sequence.lookat_points):
                    raw_lp = np.asarray(sequence.lookat_points[idx])
                    if raw_lp.shape[-1] == 3:
                        lp = raw_lp.astype(np.float64, copy=False)
                        if np.isfinite(lp).all():
                            lookat_point_ref = (
                                head_T_ref[:3, :3] @ lp.reshape(3, 1)
                                + head_T_ref[:3, 3:4]
                            ).reshape(3)
                elif derived_lookat is not None:
                    lookat_point_ref = (
                        head_T_ref[:3, :3] @ derived_lookat.reshape(3, 1)
                        + head_T_ref[:3, 3:4]
                    ).reshape(3)

                for side, resource in wrist_resources.items():
                    if idx >= len(resource["mapping"]):
                        continue
                    side_idx = resource["mapping"][idx]
                    if side_idx < 0 or side_idx >= resource["depth"].shape[0]:
                        continue
                    side_depth = resource["depth"][side_idx]
                    xyz_side = depth2xyzmap(side_depth, intrinsic)
                    mask_side = side_depth > 0
                    pts_side = xyz_side[mask_side]
                    if pts_side.size == 0:
                        continue
                    camera_pose = resource["poses"][side_idx]
                    ref_T_side = ref_T_world @ camera_pose
                    pts_ref = (
                        ref_T_side[:3, :3] @ pts_side.T + ref_T_side[:3, 3:4]
                    ).T
                    if pts_ref.size == 0:
                        continue

                    side_colors = None
                    if (
                        options.use_rgb_colors
                        and resource["rgb_frames"]
                        and resource["depth_to_rgb"] is not None
                    ):
                        rgb_idx = resource["depth_to_rgb"][side_idx]
                        if 0 <= rgb_idx < len(resource["rgb_frames"]):
                            wrist_rgb = resource["rgb_frames"][rgb_idx]
                            wrist_rgb = cv2.resize(
                                wrist_rgb,
                                (options.depth_shape[1], options.depth_shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            wrist_rgb = cv2.cvtColor(wrist_rgb, cv2.COLOR_BGR2RGB)
                            flat_rgb = (
                                wrist_rgb.reshape(-1, 3).astype(np.float32) / 255.0
                            )
                            side_colors = flat_rgb[mask_side.reshape(-1)]
                    if side_colors is None or len(side_colors) != pts_ref.shape[0]:
                        side_colors = np.tile(resource["color"], (pts_ref.shape[0], 1))
                    points_list.append(pts_ref)
                    colors_list.append(side_colors.astype(np.float64, copy=False))

                all_points = np.concatenate(points_list, axis=0)
                all_colors = np.concatenate(colors_list, axis=0)

                if len(all_points) > options.max_points:
                    choice = rng.choice(len(all_points), size=options.max_points, replace=False)
                    all_points = all_points[choice]
                    all_colors = all_colors[choice]
                elif len(all_points) == 0:
                    if options.verbose:
                        print(f"[pointmap_viz] frame {idx} skipped: all_points empty after aggregation")
                    pbar.update(1)
                    continue

                pointcloud = o3d.geometry.PointCloud()
                pointcloud.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
                pointcloud.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64))

                renderer.scene.clear_geometry()
                renderer.scene.add_geometry("pointcloud", pointcloud, point_material)

                lookat_center, eye = _compute_camera_pose(all_points, options)
                renderer.scene.camera.look_at(
                    lookat_center.tolist(), eye.tolist(), list(options.up)
                )

                reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=options.axis_length
                )
                reference_frame.compute_vertex_normals()
                renderer.scene.add_geometry("reference_frame", reference_frame, frame_material)

                extra_points = []
                extra_colors = []
                if options.save_combined_pointmap:
                    verts, cols = _mesh_vertices_with_color(reference_frame)
                    extra_points.append(verts)
                    extra_colors.append(cols)

                for side, data in gripper_data.items():
                    if len(data["poses"]) == 0:
                        continue
                    pose_idx = min(idx, len(data["poses"]) - 1)
                    if data["is_lost"][pose_idx]:
                        continue
                    world_T_tcp = data["poses"][pose_idx]
                    ref_T_tcp = ref_T_world @ world_T_tcp
                    for axis_name, template in axis_templates.items():
                        axis_mesh = _clone_geometry(template)
                        axis_mesh.transform(ref_T_tcp)
                        geom_name = f"{side}_{axis_name}"
                        renderer.scene.add_geometry(geom_name, axis_mesh, frame_material)
                        if options.save_combined_pointmap:
                            verts, cols = _mesh_vertices_with_color(axis_mesh)
                            extra_points.append(verts)
                            extra_colors.append(cols)

                if options.save_combined_pointmap and not pointmap_written:
                    if extra_points:
                        extra_points_arr = np.concatenate(extra_points, axis=0)
                        extra_colors_arr = np.concatenate(extra_colors, axis=0)
                        save_points = np.concatenate([all_points, extra_points_arr], axis=0)
                        save_colors = np.concatenate([all_colors, extra_colors_arr], axis=0)
                else:
                    save_points = all_points
                    save_colors = all_colors

                if lookat_point_ref is not None:
                    extra_vertex = lookat_point_ref.reshape(1, 3)
                    extra_color = np.array([[0.85, 0.2, 0.5]], dtype=np.float64)
                    save_points = np.concatenate([save_points, extra_vertex], axis=0)
                    save_colors = np.concatenate([save_colors, extra_color], axis=0)

                    ply_name = options.combined_pointmap_pattern.format(
                        depth_side=options.depth_side, frame=idx
                    )
                    ply_path = os.path.join(save_root, ply_name)
                    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
                    colors_uint8 = np.clip(save_colors * 255, 0, 255).astype(np.uint8)
                    with open(ply_path, "w") as f:
                        f.write("ply\nformat ascii 1.0\n")
                        f.write(f"element vertex {len(save_points)}\n")
                        f.write("property float x\nproperty float y\nproperty float z\n")
                        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                        f.write("end_header\n")
                        for pt, col in zip(save_points, colors_uint8):
                            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")
                    pointmap_written = True

                lookat_marker = o3d.geometry.TriangleMesh.create_sphere(
                    radius=options.lookat_marker_radius
                )
                lookat_marker.paint_uniform_color((0.85, 0.2, 0.5))
                marker_position = lookat_point_ref if lookat_point_ref is not None else lookat_center
                lookat_marker.translate(marker_position)
                lookat_marker.compute_vertex_normals()
                renderer.scene.add_geometry("lookat_marker", lookat_marker, frame_material)
                if options.save_combined_pointmap:
                    verts, cols = _mesh_vertices_with_color(lookat_marker)
                    extra_points.append(verts)
                    extra_colors.append(cols)

                image = renderer.render_to_image()
                frame_img = np.asarray(image)
                if frame_img.shape[2] == 4:
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
                else:
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    frame_img,
                    f"frame {idx}",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                video_out.write(frame_img)
                frames_rendered += 1
                pbar.update(1)
    except Exception as exc:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Pointmap viz internal failure for {display_name}: {exc}\n{tb}"
        ) from exc
    finally:
        video_out.release()
        if hasattr(renderer, "release"):
            renderer.release()

    if frames_rendered == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(
            f"No valid frames rendered for {display_name}; check depth data or thresholds."
        )

    return output_path

def load_replay_buffer_sequence(
    replay_buffer,
    episode_idx: int,
    options: PointmapVizOptions,
    camera_side: Optional[str] = None,
) -> PointmapSequence:
    """
    Build a PointmapSequence backed by a ReplayBuffer episode.
    """
    camera_side = camera_side or options.depth_side
    episode_slice = replay_buffer.get_episode_slice(episode_idx)

    pointmap_key = f"camera_{camera_side}_pointmap"
    depth_key = f"camera_{camera_side}_depth"
    rgb_key = f"camera_{camera_side}_main_rgb"
    lookat_key = f"camera_{camera_side}_lookatpoint"
    pos_key = f"gripper_{camera_side}_eef_pos"
    rot_key = f"gripper_{camera_side}_eef_rot_axis_angle"

    if pos_key not in replay_buffer or rot_key not in replay_buffer:
        raise KeyError(f"Replay buffer missing pose data for side '{camera_side}'")

    head_pos = np.asarray(replay_buffer[pos_key][episode_slice])
    head_rot = np.asarray(replay_buffer[rot_key][episode_slice])
    head_pose_vec = np.concatenate([head_pos, head_rot], axis=-1)
    head_poses = pose_to_mat(head_pose_vec)
    head_is_lost = np.zeros(len(head_poses), dtype=bool)

    depth_frames = None
    head_pointmaps = None
    if pointmap_key in replay_buffer:
        head_pointmaps = EpisodeArrayView(replay_buffer[pointmap_key], episode_slice)
    if depth_key in replay_buffer:
        depth_frames = EpisodeArrayView(replay_buffer[depth_key], episode_slice)
    if head_pointmaps is None and depth_frames is None:
        raise KeyError(
            f"Replay buffer is missing depth/pointmap data for camera '{camera_side}'"
        )

    rgb_frames = []
    pose_to_rgb_idx = None
    if rgb_key in replay_buffer:
        rgb_view = EpisodeArrayView(replay_buffer[rgb_key], episode_slice)
        rgb_frames = _RGBFrameAccessor(rgb_view, stored_format="rgb")
        pose_to_rgb_idx = np.arange(len(head_poses), dtype=np.int32)

    lookat_points = None
    if lookat_key in replay_buffer:
        lookat_points = EpisodeArrayView(replay_buffer[lookat_key], episode_slice)

    gripper_data: Dict[str, Dict[str, np.ndarray]] = {}
    for side in options.gripper_sides:
        g_pos_key = f"gripper_{side}_eef_pos"
        g_rot_key = f"gripper_{side}_eef_rot_axis_angle"
        if g_pos_key not in replay_buffer or g_rot_key not in replay_buffer:
            continue
        g_pos = np.asarray(replay_buffer[g_pos_key][episode_slice])
        g_rot = np.asarray(replay_buffer[g_rot_key][episode_slice])
        g_pose_vec = np.concatenate([g_pos, g_rot], axis=-1)
        g_poses = pose_to_mat(g_pose_vec)
        gripper_data[side] = {
            "poses": g_poses,
            "is_lost": np.zeros(len(g_poses), dtype=bool),
        }

    sequence = PointmapSequence(
        depth_frames=depth_frames,
        head_pointmaps=head_pointmaps,
        head_poses=head_poses,
        head_is_lost=head_is_lost,
        rgb_frames=rgb_frames,
        pose_to_rgb_idx=pose_to_rgb_idx,
        gripper_data=gripper_data,
        wrist_resources={},
        lookat_points=lookat_points,
    )
    return sequence
