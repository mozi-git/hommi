import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from hommi.common.timecode_util import datetime_fromisoformat
from hommi.common.transform_util import pos_quat_xyzw_to_4x4
from hommi.demonstration_processing.utils.generic_util import (
    get_demonstration_type,
    get_demonstration_json_data,
    read_demonstration_metadata,
)
from hommi.demonstration_processing.utils.gripper_util import iphone_to_tcp_poses


_AXIS_COLORS_BGR = (
    (0, 0, 255),  # x axis: red
    (0, 255, 0),  # y axis: green
    (255, 0, 0),  # z axis: blue
)
_ARKIT_TO_CAMERA = np.eye(4, dtype=np.float32)
_ARKIT_TO_CAMERA[1, 1] = -1.0
_ARKIT_TO_CAMERA[2, 2] = -1.0


def _load_intrinsics(calibration_path: str, camera_key: str) -> np.ndarray:
    with open(os.path.abspath(calibration_path), "r", encoding="ascii") as f:
        calib = yaml.safe_load(f)
    if camera_key not in calib:
        raise KeyError(
            f"Camera key '{camera_key}' not found in calibration file."
        )
    intrinsics = np.array(calib[camera_key]["intrinsics"], dtype=np.float32)
    return intrinsics


def _read_pose_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    poses = pos_quat_xyzw_to_4x4(
        df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values.astype(np.float32)
    )
    if "is_lost" in df.columns:
        is_lost = df["is_lost"].values.astype(bool)
    else:
        is_lost = np.zeros(len(df), dtype=bool)
    return poses, is_lost


def _get_rgb_times_key(rgb_stream: str) -> str:
    return "ultrawideRGBTimes" if rgb_stream == "ultrawide" else "rgbTimes"


def _build_pose_to_rgb_indices(
    demonstration_dir: str,
    side: str,
    n_pose_frames: int,
    rgb_stream: str,
) -> Optional[np.ndarray]:
    metadata = read_demonstration_metadata(demonstration_dir)
    pose_time_strs = metadata.get(f"{side}_frame_times")
    if not pose_time_strs:
        return None
    pose_ts = np.array(
        [datetime_fromisoformat(t).timestamp() for t in pose_time_strs[:n_pose_frames]],
        dtype=np.float64,
    )
    rgb_json = get_demonstration_json_data(demonstration_dir, side)
    rgb_times = rgb_json.get(_get_rgb_times_key(rgb_stream))
    if not rgb_times:
        return None
    rgb_ts = np.array(
        [datetime_fromisoformat(t).timestamp() for t in rgb_times], dtype=np.float64
    )
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


def _select_pose_index_for_rgb(
    pose_to_rgb_idx: Optional[np.ndarray],
    target_rgb_idx: int,
) -> int:
    if pose_to_rgb_idx is None or len(pose_to_rgb_idx) == 0:
        return max(0, target_rgb_idx)
    diffs = np.abs(pose_to_rgb_idx - target_rgb_idx)
    return int(np.argmin(diffs))


def _project_points(
    points_world: np.ndarray, camera_T_world: np.ndarray, intrinsic: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1
    )
    cam_points = (camera_T_world @ points_h.T).T[:, :3]
    z = cam_points[:, 2]
    valid = z > 1e-6
    uv = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        uv[valid, 0] = (
            intrinsic[0, 0] * cam_points[valid, 0] / z[valid] + intrinsic[0, 2]
        )
        uv[valid, 1] = (
            intrinsic[1, 1] * cam_points[valid, 1] / z[valid] + intrinsic[1, 2]
        )
    return uv, z, valid


def _draw_axes(
    image: np.ndarray,
    origin_uv: np.ndarray,
    axis_uv: np.ndarray,
    thickness: int,
) -> None:
    origin = tuple(int(x) for x in origin_uv)
    for axis_idx in range(3):
        end = tuple(int(x) for x in axis_uv[axis_idx])
        cv2.line(image, origin, end, _AXIS_COLORS_BGR[axis_idx], thickness)
    cv2.circle(image, origin, thickness + 1, (255, 255, 255), -1)


def _load_rgb_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _get_head_video_path(demo_dir: str, rgb_stream: str) -> Optional[str]:
    if rgb_stream == "ultrawide":
        candidate = os.path.join(demo_dir, "head_ultrawidergb.mp4")
        return candidate if os.path.exists(candidate) else None
    masked = os.path.join(demo_dir, "head_rgb_masked.mp4")
    if os.path.exists(masked):
        return masked
    candidate = os.path.join(demo_dir, "head_rgb.mp4")
    if os.path.exists(candidate):
        return candidate
    return None


def _load_frame_times(metadata: Dict, side: str) -> Optional[np.ndarray]:
    time_strs = metadata.get(f"{side}_frame_times")
    if not time_strs:
        return None
    return np.array(
        [datetime_fromisoformat(t).timestamp() for t in time_strs], dtype=np.float64
    )


def _resolve_rgb_frame_index(
    demonstration_dir: str,
    video_path: str,
    target_rgb_idx: int,
    rgb_stream: str,
) -> Tuple[int, Optional[int]]:
    rgb_json = get_demonstration_json_data(demonstration_dir, "head")
    rgb_times = rgb_json.get(_get_rgb_times_key(rgb_stream))
    rgb_count = len(rgb_times) if rgb_times else None
    if rgb_count is not None and rgb_count > 0:
        if target_rgb_idx < 0:
            return rgb_count - 1, rgb_count
        return min(target_rgb_idx, rgb_count - 1), rgb_count
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count > 0:
        if target_rgb_idx < 0:
            return frame_count - 1, frame_count
        return min(target_rgb_idx, frame_count - 1), frame_count
    return max(0, target_rgb_idx), None


def _render_demo(
    demo_dir: str,
    output_path: str,
    intrinsic: np.ndarray,
    axis_length: float,
    axis_thickness: int,
    gripper_sides: Tuple[str, ...],
    rotate_head_poses: bool,
    target_rgb_idx: int,
    rgb_stream: str,
) -> Optional[str]:
    video_path = _get_head_video_path(demo_dir, rgb_stream)
    if video_path is None:
        print(f"[head_rgb_axes] Missing head RGB video in {demo_dir}; skipping.")
        return None

    resolved_rgb_idx, rgb_count = _resolve_rgb_frame_index(
        demo_dir, video_path, target_rgb_idx, rgb_stream
    )
    frame = _load_rgb_frame(video_path, resolved_rgb_idx)
    if frame is None:
        print(
            f"[head_rgb_axes] Failed to read head RGB frame {resolved_rgb_idx} from {video_path}"
        )
        return None

    head_csv = os.path.join(demo_dir, "camera_trajectory_iphone_head.csv")
    if not os.path.exists(head_csv):
        print(f"[head_rgb_axes] Missing head pose CSV in {demo_dir}; skipping.")
        return None

    head_poses_arkit, head_is_lost = _read_pose_csv(head_csv)
    pose_to_rgb_idx = _build_pose_to_rgb_indices(
        demo_dir, "head", len(head_poses_arkit), rgb_stream
    )
    pose_idx = _select_pose_index_for_rgb(pose_to_rgb_idx, resolved_rgb_idx)
    if pose_to_rgb_idx is None or len(pose_to_rgb_idx) == 0:
        pose_idx = min(pose_idx, len(head_poses_arkit) - 1)
    if pose_idx >= len(head_poses_arkit):
        print(f"[head_rgb_axes] Head pose index out of range in {demo_dir}")
        return None
    if head_is_lost[pose_idx]:
        print(f"[head_rgb_axes] Head pose marked lost at frame 0 in {demo_dir}")
        return None

    head_pose_world = head_poses_arkit[pose_idx]
    if rotate_head_poses:
        z_rot = np.array(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        head_pose_world = head_pose_world @ z_rot
    camera_T_world = _ARKIT_TO_CAMERA @ np.linalg.inv(head_pose_world)

    metadata = read_demonstration_metadata(demo_dir)
    head_ts = _load_frame_times(metadata, "head")

    for side in gripper_sides:
        csv_path = os.path.join(demo_dir, f"camera_trajectory_iphone_{side}.csv")
        if not os.path.exists(csv_path):
            continue

        side_poses_arkit, side_is_lost = _read_pose_csv(csv_path)
        if len(side_poses_arkit) == 0:
            continue

        if head_ts is not None:
            side_ts = _load_frame_times(metadata, side)
            if side_ts is not None:
                mapping = _build_pose_to_side_indices(
                    head_ts, side_ts, len(side_poses_arkit)
                )
                side_idx = int(mapping[min(pose_idx, len(mapping) - 1)])
            else:
                side_idx = min(pose_idx, len(side_poses_arkit) - 1)
        else:
            side_idx = min(pose_idx, len(side_poses_arkit) - 1)

        if side_idx < 0 or side_idx >= len(side_poses_arkit):
            continue
        if side_is_lost[side_idx]:
            continue

        side_poses_tcp = iphone_to_tcp_poses(
            side_poses_arkit,
            not_use_iphone_tcp_offset=False,
            rotate_z_pi=False,
        )
        world_T_tcp = side_poses_tcp[side_idx]
        origin = world_T_tcp[:3, 3]
        axes = world_T_tcp[:3, :3]
        points_world = np.stack(
            [
                origin,
                origin + axis_length * axes[:, 0],
                origin + axis_length * axes[:, 1],
                origin + axis_length * axes[:, 2],
            ],
            axis=0,
        )
        uv, _, valid = _project_points(points_world, camera_T_world, intrinsic)
        if not valid[0]:
            continue
        axis_uv = uv[1:]
        if not np.all(valid[1:]):
            continue
        _draw_axes(frame, uv[0], axis_uv, axis_thickness)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the first head RGB frame for each episode with gripper axes overlayed."
        )
    )
    parser.add_argument("session_dir", type=str, help="Path to the session directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for rendered frames (default: <session>/viz/head_rgb_gripper_axes).",
    )
    parser.add_argument(
        "--gripper-sides",
        nargs="+",
        default=["left", "right"],
        help="Gripper sides to overlay.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Length of each axis in meters.",
    )
    parser.add_argument(
        "--axis-thickness",
        type=int,
        default=2,
        help="Line thickness in pixels.",
    )
    parser.add_argument(
        "--iphone-calibration",
        type=str,
        default="hommi/demonstration_processing/calibration/iphone15pro_calibration.yaml",
        help="Calibration YAML containing the RGB intrinsics.",
    )
    parser.add_argument(
        "--rgb-camera",
        type=str,
        default="wide",
        help="Camera key inside the calibration YAML (default: wide).",
    )
    parser.add_argument(
        "--rgb-stream",
        type=str,
        choices=["wide", "ultrawide"],
        default="wide",
        help="Head RGB stream to render (default: wide).",
    )
    parser.add_argument(
        "--rotate-head-poses",
        action="store_true",
        help="Apply the 180-degree head rotation used in dataset generation.",
    )
    parser.add_argument(
        "--rgb-frame-index",
        type=int,
        default=0,
        help="Head RGB frame index to align to (default: 0, use -1 for last frame).",
    )

    args = parser.parse_args()
    session_dir = os.path.abspath(args.session_dir)
    demos_dir = os.path.join(session_dir, "demos")
    if not os.path.isdir(demos_dir):
        raise FileNotFoundError(f"Missing demos directory at {demos_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(session_dir, "viz", "head_rgb_gripper_axes")
    output_dir = os.path.abspath(output_dir)

    rgb_camera = args.rgb_camera
    if args.rgb_stream == "ultrawide" and rgb_camera == "wide":
        rgb_camera = "ultrawide"
    intrinsic = _load_intrinsics(args.iphone_calibration, rgb_camera)

    rendered = 0
    for demo_name in sorted(os.listdir(demos_dir)):
        demo_dir = os.path.join(demos_dir, demo_name)
        if not os.path.isdir(demo_dir):
            continue
        if get_demonstration_type(demo_dir) != "demonstration":
            continue

        output_path = os.path.join(output_dir, f"{demo_name}.png")
        result = _render_demo(
            demo_dir=demo_dir,
            output_path=output_path,
            intrinsic=intrinsic,
            axis_length=args.axis_length,
            axis_thickness=args.axis_thickness,
            gripper_sides=tuple(args.gripper_sides),
            rotate_head_poses=args.rotate_head_poses,
            target_rgb_idx=args.rgb_frame_index,
            rgb_stream=args.rgb_stream,
        )
        if result is not None:
            rendered += 1
            print(f"[head_rgb_axes] Wrote {result}")

    if rendered == 0:
        print("[head_rgb_axes] No frames rendered.")


if __name__ == "__main__":
    main()
