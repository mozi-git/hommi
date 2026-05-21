"""Streaming bridge between the policy server and the realtime controller."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent
HOMMI_ROOT = PROJECT_ROOT.parents[1]
RBY1_WBC_ROOT = HOMMI_ROOT / "deps" / "rby1-wbc"
for path in (HOMMI_ROOT, RBY1_WBC_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
PROJECT_ROOT = str(PROJECT_ROOT)
RBY1_WBC_ROOT = str(RBY1_WBC_ROOT)

from rby1.frame_transforms import MODEL_TO_TCP_FRAME, TCP_TO_MODEL_FRAME, apply_transform_tf
from hommi.deployment.camera.camera_stream import AravisCameraStreamer
from hommi.deployment.rby1_policy_robot import RBY1PolicyRobot, ScheduledAction


DEFAULT_CAMERA_LATENCIES = {
    # Wrist cameras (left/right)
    "camera_left_main_rgb": 0.1,
    "camera_right_main_rgb": 0.1,
    # Head rig (main, right, ultrawide)
    "camera_head_main_rgb": 0.06,
    "camera_head_main_right_rgb": 0.06,
    "camera_head_ultrawide_rgb": 0.1,
}


@dataclass
class DebugActionRequest:
    actions: Sequence[ScheduledAction]
    decision_event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    current_pose: Optional[Dict[str, np.ndarray]] = None
    image_obs: Optional[Dict[str, np.ndarray]] = None

    def resolve(self, decision: bool) -> None:
        self.approved = decision
        self.decision_event.set()


@dataclass
class DebugImageRequest:
    image_obs: Dict[str, np.ndarray]
    output_dir: Path
    batch_tag: str
    bgr_to_rgb: bool


@dataclass
class DebugEnvObsRequest:
    env_obs: Dict[str, np.ndarray]
    output_dir: Path
    batch_tag: str


OBS_TF_KEYS = {
    "left": "gripper_left_tf",
    "right": "gripper_right_tf",
    "head": "head_tf",
}
GRIPPER_WIDTH_LIMITS = (0.0, 0.085)


def _convert_robot_observations(
    robot_obs: Dict[str, np.ndarray],
    transform_map: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in robot_obs.items()}
    for effector, key in OBS_TF_KEYS.items():
        if key not in robot_obs or effector not in transform_map:
            continue
        converted[key] = apply_transform_tf(robot_obs[key], transform_map[effector])
    return converted


def _offset_gripper_obs(
    obs: Dict[str, np.ndarray],
    offset: float,
) -> Dict[str, np.ndarray]:
    if offset == 0.0:
        return obs
    adjusted = dict(obs)
    for key in ("gripper_left_gripper_width", "gripper_right_gripper_width"):
        if key in adjusted:
            adjusted[key] = np.asarray(adjusted[key], dtype=float) + offset
    return adjusted

def _merge_camera_timestamps(camera_ts: Dict[str, np.ndarray]) -> np.ndarray:
    """Use the slowest camera as the reference observation clock."""

    if not camera_ts:
        raise ValueError("camera_ts must contain at least one camera stream")
    arrays = [np.asarray(ts, dtype=float).reshape(-1) for ts in camera_ts.values()]
    reference = arrays[0]
    for arr in arrays[1:]:
        if arr.shape != reference.shape:
            raise ValueError("Camera timestamp arrays must share the same shape")
        reference = np.maximum(reference, arr)
    return reference


def _parse_latency_overrides(entries: Optional[Sequence[str]]) -> Dict[str, float]:
    if not entries:
        return {}
    mapping: Dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid camera latency override '{entry}', expected KEY=SECONDS")
        key, val = entry.split("=", 1)
        key = key.strip()
        try:
            mapping[key] = float(val)
        except ValueError:
            raise ValueError(f"Invalid latency value in override '{entry}'") from None
    return mapping


def _camera_latency_for(key: str, global_override: Optional[float], overrides: Dict[str, float]) -> float:
    if global_override is not None:
        return float(global_override)
    if key in overrides:
        return float(overrides[key])
    return float(DEFAULT_CAMERA_LATENCIES.get(key, 0.0))


def _derive_required_env_obs_keys(obs_meta: Dict[str, object]) -> set[str]:
    """Translate policy observation meta into the raw observation keys the server expects."""

    required: set[str] = set()
    if not isinstance(obs_meta, dict):
        return required

    for key, attr in obs_meta.items():
        if not isinstance(attr, dict):
            continue
        obs_type = attr.get("type", "low_dim")
        if obs_type in {"rgb", "pointmap"}:
            required.add(key)
        elif obs_type == "low_dim" and "eef" not in key:
            required.add(key)

        if "_eef_pos" in key:
            prefix = key.split("_eef_pos")[0]
            required.add(f"{prefix}_tf")
            width_key = f"{prefix}_gripper_width"
            if width_key in obs_meta:
                required.add(width_key)
        if "_eef_rot_axis_angle" in key:
            prefix = key.split("_eef_rot_axis_angle")[0]
            required.add(f"{prefix}_tf")

        # Head frame is needed to transform pointmaps / lookatpoint into the requested frame
        if obs_type == "pointmap" or key == "camera_head_lookatpoint":
            required.add("head_tf")
        if obs_type == "pointmap":
            required.add(key.replace("_pointmap", "_main_rgb"))
            required.add(key.replace("_pointmap", "_main_right_rgb"))

    return required

class PolicyClient:
    """Thin ZMQ wrapper that talks to the detached policy server."""

    def __init__(self, ip: str, port: int, timeout: float = 2.0) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{ip}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, int(max(timeout, 0.1) * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(max(timeout, 0.1) * 1000))
        self.observation_keys: Dict[str, object] = {}

    def close(self) -> None:
        self._socket.close(0)

    def request_observation_keys(self) -> Dict[str, object]:
        while True:
            try:
                self._socket.send_string("get_obs_keys")
                reply = self._socket.recv_pyobj()
                # print(f"[policy] Received observation keys: {reply}")
            except zmq.Again:
                continue
            if isinstance(reply, dict):
                self.observation_keys = reply
                return reply
            time.sleep(1.0)

    def infer(self, obs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        try:
            self._socket.send_pyobj(obs)
            return self._socket.recv_pyobj()
        except (zmq.Again, zmq.ZMQError):
            return None


def build_scheduled_actions(
    actions_tf: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    fallback_dt: float,
    now: float,
    gripper_width_offset: float,
) -> List[ScheduledAction]:
    if not actions_tf:
        return []

    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)

    def _timestamp_for_index(idx: int) -> float:
        if timestamps.size == 0:
            return now + fallback_dt * (idx + 1)
        idx = min(idx, timestamps.size - 1)
        ts = float(timestamps[idx])
        if not np.isfinite(ts):
            return now + fallback_dt * (idx + 1)
        return ts

    eef_length = max(
        (
            np.asarray(actions_tf[key]).shape[0]
            for key in ("gripper_left_tf", "gripper_right_tf", "head_tf", "camera_head_lookatpoint")
            if key in actions_tf and actions_tf[key] is not None
        ),
        default=0,
    )
    gripper_length = max(
        (
            np.asarray(actions_tf[key]).shape[0]
            for key in ("gripper_left_gripper_width", "gripper_right_gripper_width")
            if key in actions_tf and actions_tf[key] is not None
        ),
        default=0,
    )

    if eef_length == 0 and gripper_length == 0:
        return []

    scheduled: List[ScheduledAction] = []
    if eef_length > 0:
        for idx in range(eef_length):
            payload: Dict[str, np.ndarray] = {}
            for effector, obs_key in OBS_TF_KEYS.items():
                payload_key = obs_key
                if effector == "head" and "camera_head_lookatpoint" in actions_tf:
                    payload_key = "camera_head_lookatpoint"
                elif effector == "head" and "head_tf" in actions_tf:
                    payload_key = "gripper_head_tf"
                if payload_key not in actions_tf:
                    continue
                series = np.asarray(actions_tf[payload_key])
                if idx >= series.shape[0]:
                    continue
                value = series[idx]
                if effector in TCP_TO_MODEL_FRAME and (payload_key.endswith("_tf")):
                    value = apply_transform_tf(value, TCP_TO_MODEL_FRAME[effector])
                payload[payload_key] = value
            if not payload:
                continue

            timestamp = _timestamp_for_index(idx)
            if idx + 1 < eef_length:
                next_timestamp = _timestamp_for_index(idx + 1)
                duration = float(max(fallback_dt, next_timestamp - timestamp))
            else:
                duration = float(fallback_dt)

            scheduled.append(ScheduledAction(timestamp=timestamp, duration=duration, payload=payload))

    if gripper_length > 0:
        for idx in range(gripper_length):
            payload: Dict[str, np.ndarray] = {}
            if "gripper_left_gripper_width" in actions_tf:
                width_series = np.asarray(actions_tf["gripper_left_gripper_width"], dtype=float)
                if idx < width_series.shape[0]:
                    payload["gripper_left_gripper_width"] = np.clip(
                        float(width_series[idx].reshape(-1)[0]) - gripper_width_offset,
                        GRIPPER_WIDTH_LIMITS[0],
                        GRIPPER_WIDTH_LIMITS[1],
                    )
                    # print(f"gripper width offset: {gripper_width_offset}")  # pragma: no cover - debug aid
                    # print(f"[policy] gripper left width at idx {idx}: {payload['gripper_left_gripper_width']}")
            if "gripper_right_gripper_width" in actions_tf:
                width_series = np.asarray(actions_tf["gripper_right_gripper_width"], dtype=float)
                if idx < width_series.shape[0]:
                    payload["gripper_right_gripper_width"] = np.clip(
                        float(width_series[idx].reshape(-1)[0]) - gripper_width_offset,
                        GRIPPER_WIDTH_LIMITS[0],
                        GRIPPER_WIDTH_LIMITS[1],
                    )
                    # print(f"[policy] gripper right width at idx {idx}: {payload['gripper_right_gripper_width']}")

            if not payload:
                continue

            timestamp = _timestamp_for_index(idx)
            if idx + 1 < gripper_length:
                next_timestamp = _timestamp_for_index(idx + 1)
                duration = float(max(fallback_dt, next_timestamp - timestamp))
            else:
                duration = float(fallback_dt)

            scheduled.append(ScheduledAction(timestamp=timestamp, duration=duration, payload=payload))

    if not scheduled:
        return []

    scheduled.sort(key=lambda a: a.timestamp)
    return scheduled

def _plot_action_chunk(
    actions: Sequence[ScheduledAction],
    current_pose: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Callable[[], None]]:
    """Visualize scheduled actions. Returns a cleanup callback if plotting succeeded."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import cm as _cm
        from matplotlib import colors as _colors
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed for 3D projection registration
    except Exception as exc:  # pragma: no cover - debug-only helper
        print(f"[debug] Unable to plot actions (matplotlib missing?): {exc}")
        return None

    if not actions:
        return None
    ee_trajectories: Dict[str, np.ndarray] = {}
    ee_orientations: Dict[str, np.ndarray] = {}
    lookat_points: List[np.ndarray] = []
    for name, payload_key in OBS_TF_KEYS.items():
        coords: List[np.ndarray] = []
        rots: List[np.ndarray] = []
        for action in actions:
            payload = action.payload
            if payload_key not in payload:
                continue
            tf = np.asarray(payload[payload_key], dtype=float).reshape(4, 4)
            coords.append(tf[:3, 3])
            rots.append(Rotation.from_matrix(tf[:3, :3]).as_rotvec())
        if coords:
            ee_trajectories[name] = np.vstack(coords)
            ee_orientations[name] = np.vstack(rots)

    for action in actions:
        payload = action.payload
        if "camera_head_lookatpoint" in payload:
            lookat_points.append(np.asarray(payload["camera_head_lookatpoint"], dtype=float).reshape(3))

    other_keys = sorted({key for action in actions for key in action.payload})
    excluded_keys = set(OBS_TF_KEYS.values())
    excluded_keys.update([f"{name}_rot_axis_angle" for name in OBS_TF_KEYS])
    plot_data: List[Tuple[str, np.ndarray]] = []
    for key in other_keys:
        if key in excluded_keys:
            continue
        values: List[np.ndarray] = []
        for action in actions:
            payload = action.payload
            if key not in payload:
                continue
            value = np.asarray(payload[key], dtype=float).reshape(-1)
            values.append(value)
        if values:
            stacked = np.vstack(values)
            if stacked.ndim == 1:
                stacked = stacked[:, None]
            plot_data.append((key, stacked))

    def _set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(limits, axis=1)
        radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
        for center, setter in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            setter(center - radius, center + radius)


    figures: List[plt.Figure] = []
    if ee_trajectories:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        cmap_cycle = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        for idx, (name, trajectory) in enumerate(ee_trajectories.items()):
            cmap = _cm.get_cmap(cmap_cycle[idx % len(cmap_cycle)])
            norm = _colors.Normalize(vmin=0, vmax=max(len(trajectory) - 1, 1))
            # Plot gradient scatter along the trajectory.
            for step, pos in enumerate(trajectory):
                color = cmap(norm(step))
                ax.scatter(pos[0], pos[1], pos[2], color=color, s=40)
                rot_samples = ee_orientations.get(name)
                if rot_samples is not None and step < len(rot_samples):
                    orientation = Rotation.from_rotvec(rot_samples[step])
                    axes_dirs = orientation.apply(np.eye(3))
                    axis_colors = ["r", "g", "b"]
                    axis_scale = 0.05
                    for axis_vec, axis_color in zip(axes_dirs, axis_colors):
                        direction = axis_vec * axis_scale
                        ax.quiver(
                            pos[0],
                            pos[1],
                            pos[2],
                            direction[0],
                            direction[1],
                            direction[2],
                            color=axis_color,
                            length=0.3,
                            normalize=False,
                            arrow_length_ratio=0.2,
                        )
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=cmap(0.6),
                alpha=0.6,
                label=f"{name} traj",
            )
            if current_pose and name in current_pose:
                cur_tf = np.asarray(current_pose[name], dtype=float).reshape(4, 4)
                cur = cur_tf[:3, 3]
                ax.scatter(
                    cur[0],
                    cur[1],
                    cur[2],
                    color=cmap(1.0),
                    s=80,
                    marker="X",
                    label=f"{name} current",
                )
                cur_axes = Rotation.from_matrix(cur_tf[:3, :3]).apply(np.eye(3))
                axis_scale = 0.06
                for axis_vec, axis_color in zip(cur_axes, ["r", "g", "b"]):
                    vec = axis_vec * axis_scale
                    ax.quiver(
                        cur[0],
                        cur[1],
                        cur[2],
                        vec[0],
                        vec[1],
                        vec[2],
                        color=axis_color,
                        length=0.3,
                        normalize=False,
                        arrow_length_ratio=0.2,
                    )
            ax.text(
                trajectory[-1, 0],
                trajectory[-1, 1],
                trajectory[-1, 2],
                f"{name} end",
                color="black",
                fontsize=8,
            )
        ax.set_title("Predicted gripper trajectories")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        if lookat_points:
            lookat_arr = np.vstack(lookat_points)
            ax.scatter(
                lookat_arr[:, 0],
                lookat_arr[:, 1],
                lookat_arr[:, 2],
                color="c",
                marker="*",
                s=60,
                alpha=0.8,
                label="head lookat",
            )
            head_traj = ee_trajectories.get("head")
            if head_traj is not None and len(head_traj) and len(lookat_arr):
                seg_len = min(len(head_traj), len(lookat_arr))
                for idx in range(seg_len):
                    start = head_traj[idx]
                    end = lookat_arr[idx]
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color="c",
                        linestyle="--",
                        alpha=0.6,
                    )
        ax.legend(loc="best")
        _set_axes_equal(ax)
        figures.append(fig)

    if not figures:
        return None

    plt.show(block=False)
    plt.pause(0.01)

    def _cleanup() -> None:
        for figure in figures:
            plt.close(figure)

    return _cleanup


def _plot_image_observations(
    image_obs: Optional[Dict[str, np.ndarray]],
    bgr_to_rgb: bool = True,
) -> Optional[Callable[[], None]]:
    if not image_obs:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[debug] Unable to plot image observations: {exc}")
        return None

    keys = list(image_obs.keys())
    if not keys:
        return None

    num = len(keys)
    cols = min(3, num)
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for idx, key in enumerate(keys):
        ax = axes[idx // cols, idx % cols]
        frames = np.asarray(image_obs[key])
        frame = frames[-1]
        if bgr_to_rgb and frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame[..., ::-1]
        if frame.ndim == 3 and frame.shape[-1] in (1, 3):
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0.0, 1.0)
            if frame.shape[-1] == 1:
                ax.imshow(frame[..., 0], cmap="gray")
            else:
                ax.imshow(frame)
        else:
            ax.imshow(frame, cmap="gray")
        ax.set_title(key)
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    def _cleanup() -> None:
        plt.close(fig)

    return _cleanup


def _sanitize_camera_key(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key)


def _normalize_image_frames(frames: np.ndarray, *, bgr_to_rgb: bool) -> np.ndarray:
    if frames.ndim == 3:
        frames = frames[:, :, :, None]
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T, H, W, C), got {frames.shape}")

    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif frames.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {frames.shape[-1]}")

    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames, 0.0, 1.0) * 255.0
    else:
        frames = np.clip(frames, 0, 255)
    if bgr_to_rgb:
        frames = frames[..., ::-1]
    return frames.astype(np.uint8, copy=False)


def _save_debug_images(request: DebugImageRequest) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[debug] Unable to save images (cv2 missing): {exc}")
        return

    request.output_dir.mkdir(parents=True, exist_ok=True)
    for key, frames in request.image_obs.items():
        safe_key = _sanitize_camera_key(key)
        try:
            image_frames = _normalize_image_frames(np.asarray(frames), bgr_to_rgb=request.bgr_to_rgb)
        except ValueError as exc:
            print(f"[debug] Skipping images for '{key}': {exc}")
            continue
        for idx, frame in enumerate(image_frames):
            output_path = request.output_dir / f"{request.batch_tag}_{safe_key}_frame_{idx:04d}.png"
            bgr = frame[..., ::-1]
            if not cv2.imwrite(str(output_path), bgr):
                print(f"[debug] Failed to write image '{output_path}'")


def _save_debug_env_obs(request: DebugEnvObsRequest) -> None:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = request.output_dir / f"{request.batch_tag}_env_obs.npz"
    try:
        np.savez_compressed(output_path, **request.env_obs)
    except Exception as exc:  # pragma: no cover - best effort saver
        print(f"[debug] Failed to write env obs {output_path}: {exc}")


def _start_debug_env_obs_worker(
    max_queue: int = 4,
) -> tuple["queue.Queue[Optional[DebugEnvObsRequest]]", threading.Thread, threading.Event]:
    env_queue: "queue.Queue[Optional[DebugEnvObsRequest]]" = queue.Queue(maxsize=max_queue)
    stop_event = threading.Event()

    def _worker() -> None:
        while True:
            if stop_event.is_set() and env_queue.empty():
                break
            try:
                request = env_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                if stop_event.is_set():
                    break
                continue
            try:
                _save_debug_env_obs(request)
            except Exception as exc:  # pragma: no cover - best effort saver
                print(f"[debug] Failed to save env obs: {exc}")

    thread = threading.Thread(target=_worker, name="debug-env-obs-writer", daemon=True)
    thread.start()
    return env_queue, thread, stop_event


def _ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _start_debug_video_logger(
    camera_streamer: AravisCameraStreamer,
    output_dir: Path,
    fps: Optional[float],
    stop_event: threading.Event,
    poll_period: float = 0.01,
) -> threading.Thread:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[debug] Unable to save videos (cv2 missing): {exc}")
        return threading.Thread(target=lambda: None, name="debug-video-writer", daemon=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_fps = float(fps) if fps is not None else float(getattr(camera_streamer, "_frame_rate", 30.0))
    last_timestamps: Dict[str, float] = {}
    writers: Dict[str, cv2.VideoWriter] = {}

    def _writer_for(key: str, frame: np.ndarray) -> cv2.VideoWriter:
        if key in writers:
            return writers[key]
        h, w = frame.shape[:2]
        w_even = _ensure_even(w)
        h_even = _ensure_even(h)
        path = output_dir / f"{_sanitize_camera_key(key)}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (w_even, h_even),
        )
        writers[key] = writer
        return writer

    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        w_even = _ensure_even(w)
        h_even = _ensure_even(h)
        if (h, w) == (h_even, w_even):
            return frame
        canvas = np.zeros((h_even, w_even, 3), dtype=frame.dtype)
        canvas[:h, :w] = frame
        return canvas

    def _worker() -> None:
        while not stop_event.is_set():
            frames_by_cam = camera_streamer.get_frames_since(last_timestamps)
            wrote_any = False
            for key, frames in frames_by_cam.items():
                if not frames:
                    continue
                writer = _writer_for(key, frames[0].image)
                for frame in frames:
                    writer.write(_normalize_frame(frame.image))
                    last_timestamps[key] = frame.timestamp
                wrote_any = True
            if not wrote_any:
                stop_event.wait(timeout=poll_period)
        for writer in writers.values():
            try:
                writer.release()
            except Exception:
                pass

    thread = threading.Thread(target=_worker, name="debug-video-writer", daemon=True)
    thread.start()
    return thread


def _start_debug_image_worker(
    bgr_to_rgb: bool = True,
    max_queue: int = 4,
) -> tuple["queue.Queue[Optional[DebugImageRequest]]", threading.Thread, threading.Event]:
    image_queue: "queue.Queue[Optional[DebugImageRequest]]" = queue.Queue(maxsize=max_queue)
    stop_event = threading.Event()

    def _worker() -> None:
        while True:
            if stop_event.is_set() and image_queue.empty():
                break
            try:
                request = image_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                if stop_event.is_set():
                    break
                continue
            try:
                _save_debug_images(request)
            except Exception as exc:  # pragma: no cover - best effort saver
                print(f"[debug] Failed to save debug images: {exc}")

    thread = threading.Thread(target=_worker, name="debug-image-writer", daemon=True)
    thread.start()
    return image_queue, thread, stop_event


def _confirm_action_execution(
    actions: Sequence[ScheduledAction],
    current_pose: Optional[Dict[str, np.ndarray]] = None,
    image_obs: Optional[Dict[str, np.ndarray]] = None,
) -> bool:
    """Ask the operator for approval before sending a batch of actions to the robot."""

    cleanup_actions = _plot_action_chunk(actions, current_pose=current_pose)
    cleanup_images = _plot_image_observations(image_obs)
    plotted = cleanup_actions is not None or cleanup_images is not None
    if not sys.stdin or not sys.stdin.isatty():
        print("[debug] No interactive terminal available; skipping action execution.")
        return False

    prompt = "[debug] Execute plotted actions on the robot? [y/N]: " if plotted else (
        "[debug] Execute actions on the robot? [y/N]: "
    )
    decision: Optional[bool] = None
    try:
        while decision is None:
            try:
                response = input(prompt)
            except EOFError:
                print("[debug] Input stream closed; skipping action execution.")
                return False
            normalized = response.strip().lower()
            if not normalized:
                decision = False
            elif normalized in {"y", "yes"}:
                decision = True
            elif normalized in {"n", "no"}:
                decision = False
            else:
                print("[debug] Please respond with 'y' or 'n'.")
    finally:
        if cleanup_actions is not None:
            cleanup_actions()
        if cleanup_images is not None:
            cleanup_images()

    return bool(decision)


def _handle_debug_requests(action_queue: "queue.Queue[DebugActionRequest]") -> None:
    while True:
        try:
            request = action_queue.get_nowait()
        except queue.Empty:
            break
        try:
            approved = _confirm_action_execution(
                request.actions,
                current_pose=request.current_pose,
                image_obs=request.image_obs,
            )
        except BaseException:
            request.resolve(False)
            raise
        else:
            request.resolve(approved)


def _reject_pending_debug_requests(action_queue: "queue.Queue[DebugActionRequest]") -> None:
    while True:
        try:
            request = action_queue.get_nowait()
        except queue.Empty:
            break
        request.resolve(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 policy streaming bridge")
    parser.add_argument("--policy-ip", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8766)
    parser.add_argument("--robot-horizon", type=int, default=2)
    parser.add_argument("--camera-horizon", type=int, default=2)
    parser.add_argument("--state-only", action="store_true", help="Skip camera streaming")
    parser.add_argument("--camera-config", default=None, help="Path to camera YAML. Defaults to HOMMI_CAMERA_CONFIG or source checkout config.")
    parser.add_argument("--control-dt", type=float, default=0.1)
    parser.add_argument(
        "--policy-interval",
        type=float,
        default=0.8,
        help="Target interval between policy inference calls in seconds.",
    )
    parser.add_argument("--policy-timeout", type=float, default=2.0, help="ZMQ request timeout in seconds")
    parser.add_argument("--executor-lookahead", type=float, default=0.0, help="Execution loop lookahead in seconds")
    parser.add_argument(
        "--arm-execution-latency",
        type=float,
        default=0.2753,
        help="Measured execution latency (seconds) for the arm controller.",
    )
    parser.add_argument(
        "--gripper-execution-latency",
        type=float,
        default=0.1,    # todo: maybe need to tune
        help="Measured execution latency (seconds) for the gripper hardware.",
    )
    parser.add_argument(
        "--camera-latency",
        type=float,
        default=None,
        help="Global camera latency (seconds) applied to all cameras; defaults to per-camera values if unset.",
    )
    parser.add_argument(
        "--camera-latency-override",
        action="append",
        help=(
            "Per-camera latency override KEY=SECONDS; repeatable. "
        ),
    )
    parser.add_argument(
        "--proprioception-latency",
        type=float,
        default=0.005,
        help="Measured proprioception latency (seconds) for observation alignment.",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="Plot action batches and require manual confirmation before execution.",
    )
    parser.add_argument(
        "--debug-save-images",
        action="store_true",
        help="Save camera images for each action batch (independent of --debug-actions).",
    )
    parser.add_argument(
        "--debug-save-video",
        action="store_true",
        help="Save continuous camera videos without blocking policy inference.",
    )
    parser.add_argument(
        "--debug-video-fps",
        type=float,
        default=None,
        help="FPS for debug videos (default: camera frame rate).",
    )
    parser.add_argument(
        "--debug-video-dir",
        default="log/debug_action_videos",
        help="Output directory for debug action images/videos.",
    )
    parser.add_argument("--align-first-action", action="store_true", help="Rigidly align each action chunk to the current robot pose before execution.")
    parser.add_argument("--sim-only", action="store_true", help="Run without the realtime controller and preview actions in Mujoco.")
    parser.add_argument("--sim-model", default=None, help="Optional custom MJCF path for --sim-only mode.")
    parser.add_argument("--sim-viewer", action="store_true", help="Open a Mujoco viewer when using --sim-only.")
    parser.add_argument("--gripper-width-offset", type=float, default=0.0, help="Additive offset applied to observed gripper widths.")
    parser.add_argument(
        "--gripper-width-command-offset",
        type=float,
        default=0.005,
        help="Offset subtracted from commanded gripper widths before clipping.",
    )
    parser.add_argument(
        "--plot-tracking",
        action="store_true",
        help="Cache commanded actions and robot states and save a tracking error plot on exit.",
    )
    args = parser.parse_args()

    camera_latency_overrides = _parse_latency_overrides(args.camera_latency_override)
    control_dt = max(args.control_dt, 1e-2)
    inference_period = max(args.policy_interval, 1e-2)

    robot = RBY1PolicyRobot(
        config_path=RBY1_WBC_ROOT + "/config/wbc.yaml",
        use_sim=args.sim_only,
        sim_model_path=args.sim_model,
        sim_viewer=args.sim_viewer,
    )
    robot.start()
    camera_streamer = None
    stop_event = threading.Event()
    threads: List[threading.Thread] = []
    policy_client: Optional[PolicyClient] = None
    debug_request_queue: Optional["queue.Queue[DebugActionRequest]"] = (
        queue.Queue() if args.debug_actions else None
    )
    debug_image_queue: Optional["queue.Queue[Optional[DebugImageRequest]]"] = None
    debug_image_thread: Optional[threading.Thread] = None
    debug_image_stop: Optional[threading.Event] = None
    debug_env_obs_queue: Optional["queue.Queue[Optional[DebugEnvObsRequest]]"] = None
    debug_env_obs_thread: Optional[threading.Thread] = None
    debug_env_obs_stop: Optional[threading.Event] = None
    debug_video_thread: Optional[threading.Thread] = None
    debug_image_counter = [0]
    debug_media_root: Optional[Path] = None
    tracking_cmd: List[Tuple[float, Dict[str, np.ndarray]]] = []
    tracking_policy_raw: List[Tuple[float, Dict[str, np.ndarray]]] = []
    tracking_exec: List[Tuple[float, Dict[str, np.ndarray]]] = []
    tracking_state: List[Tuple[float, Dict[str, np.ndarray]]] = []
    tracking_state_thread: Optional[threading.Thread] = None
    latency_samples: List[Dict[str, float]] = []

    try:
        robot.wait_until_ready()
        if args.plot_tracking:
            robot.set_execution_hook(lambda ts, payload: tracking_exec.append((ts, payload)))
        required_robot_samples = max(int(args.robot_horizon * control_dt / (robot.dt)), 20)
        if not robot.wait_for_observations(required_robot_samples, timeout=5.0):
            raise TimeoutError("Timed out waiting for initial robot observations")

        if args.plot_tracking and not args.sim_only:
            def _state_sampler() -> None:
                # Sample at roughly the controller rate to overlay with commands.
                period = max(robot.dt, 0.1)
                while not stop_event.is_set():
                    obs = robot.get_latest_observation()
                    if obs is not None:
                        payload = {
                            "gripper_left_tf": obs.left_tf.copy(),
                            "gripper_right_tf": obs.right_tf.copy(),
                            "gripper_left_gripper_width": np.array([obs.left_width], dtype=float),
                            "gripper_right_gripper_width": np.array([obs.right_width], dtype=float),
                        }
                        backend = getattr(robot, "_backend", None)
                        if backend is not None and hasattr(backend, "get_latest_robot_state"):
                            snapshot = backend.get_latest_robot_state()
                            if snapshot is not None and getattr(snapshot, "is_valid", True) and hasattr(snapshot, "odom_SE2"):
                                payload["odom_SE2"] = np.asarray(snapshot.odom_SE2, dtype=float).copy()
                        tracking_state.append((obs.timestamp, payload))
                    if stop_event.wait(timeout=period):
                        break
            tracking_state_thread = threading.Thread(target=_state_sampler, name="tracking-state", daemon=True)
            tracking_state_thread.start()

        if not args.state_only and not args.sim_only:
            camera_streamer = AravisCameraStreamer(config_path=args.camera_config)
            camera_streamer.start()
            try:
                camera_streamer.wait_until_ready(
                    min_frames=max(args.camera_horizon * control_dt / camera_streamer._frame_rate, 1),
                    timeout=2.0,
                )
            except TimeoutError as exc:
                print(f"[camera] {exc}")
        elif args.sim_only and not args.state_only:
            print("[camera] Skipping camera streamer in simulation mode.")

        if args.debug_save_images or args.debug_save_video:
            debug_media_root = Path(args.debug_video_dir) / time.strftime("%Y%m%d_%H%M%S")
        if args.debug_save_images:
            debug_image_queue, debug_image_thread, debug_image_stop = _start_debug_image_worker(
                bgr_to_rgb=True,
            )
            debug_env_obs_queue, debug_env_obs_thread, debug_env_obs_stop = _start_debug_env_obs_worker()
        if args.debug_save_video and camera_streamer is not None:
            output_dir = debug_media_root if debug_media_root is not None else Path(args.debug_video_dir)
            debug_video_thread = _start_debug_video_logger(
                camera_streamer=camera_streamer,
                output_dir=output_dir,
                fps=args.debug_video_fps,
                stop_event=stop_event,
            )
        elif args.debug_save_video:
            print("[debug] Video logging requested but camera streamer is unavailable.")

        policy_client = PolicyClient(
            ip=args.policy_ip,
            port=args.policy_port,
            timeout=args.policy_timeout,
        )
        _ = policy_client.request_observation_keys()
        policy_obs_meta = policy_client.observation_keys.get("obs", {}) if isinstance(policy_client.observation_keys, dict) else {}
        required_env_obs_keys = _derive_required_env_obs_keys(policy_obs_meta)
        if required_env_obs_keys:
            print(f"[policy] Policy server requires obs keys: {sorted(required_env_obs_keys)}")
        # print(f"[policy] Required observation keys: {policy_client.observation_keys}")

        def inference_worker() -> None:
            while not stop_event.is_set():
                start = time.monotonic()
                anchor_timestamps: Optional[np.ndarray] = None
                # try:
                #     robot_obs_model = robot.get_observation_window(
                #         horizon=args.robot_horizon,
                #         stride=args.robot_stride,
                #         obs_frequency=args.obs_frequency,
                #     )
                # except Exception as exc:  # pragma: no cover - runtime safeguard
                #     print(f"[robot] Failed to gather observation: {exc}")
                #     if stop_event.wait(timeout=control_dt):
                #         break
                #     continue

                debug_images: Optional[Dict[str, np.ndarray]] = None
                camera_timestamps: Dict[str, np.ndarray] = {}
                if camera_streamer is not None:
                    try:
                        camera_result = camera_streamer.get_observation_window(
                            horizon=args.camera_horizon,
                            step=control_dt,
                            include_timestamps=True,
                        )
                        camera_obs, camera_timestamps = camera_result
                        debug_images = {k: np.asarray(v).copy() for k, v in camera_obs.items()}
                    except RuntimeError as exc:
                        print(f"[camera] {exc}")
                        camera_obs = None
                else:
                    camera_obs = None

                if camera_obs is not None and camera_timestamps:
                    adjusted_camera_ts = {
                        key: np.asarray(ts, dtype=float)
                        - _camera_latency_for(key, args.camera_latency, camera_latency_overrides)
                        for key, ts in camera_timestamps.items()
                    }
                    anchor_timestamps = _merge_camera_timestamps(adjusted_camera_ts)
                    query_timestamps = anchor_timestamps + float(args.proprioception_latency)
                    try:
                        robot_obs_model = robot.sample_observations_at(query_timestamps)
                    except Exception as exc:  # pragma: no cover - runtime safeguard
                        print(f"[robot] Failed to align observations: {exc}")
                        if stop_event.wait(timeout=control_dt):
                            break
                        continue
                else:
                    anchor_timestamps = np.asarray(robot_obs_model.get("timestamp", []), dtype=float) - float(
                        args.proprioception_latency
                    )

                policy_robot_obs = _convert_robot_observations(robot_obs_model, MODEL_TO_TCP_FRAME)
                policy_robot_obs = _offset_gripper_obs(policy_robot_obs, args.gripper_width_offset)
                if "head_tf" in policy_robot_obs and "gripper_head_tf" not in policy_robot_obs:
                    # Alias head_tf for policies expecting gripper_head_tf.
                    policy_robot_obs["gripper_head_tf"] = np.asarray(policy_robot_obs["head_tf"]).copy()
                obs_dict = {k: v for k, v in policy_robot_obs.items() if k != "timestamp"}
                if camera_obs is not None:
                    for key, value in camera_obs.items():
                        if required_env_obs_keys and key not in required_env_obs_keys:
                            continue
                        obs_dict[key] = value

                obs_dict["timestamp"] = anchor_timestamps
                if required_env_obs_keys:
                    missing_obs_keys = [k for k in required_env_obs_keys if k not in obs_dict]
                    if missing_obs_keys:
                        print(f"[policy] Missing required observation keys from policy server: {missing_obs_keys}")
                        if stop_event.wait(timeout=control_dt):
                            break
                        continue
                # Policy inference
                infer_start = time.monotonic()
                reply = policy_client.infer(obs_dict)
                infer_end = time.monotonic()
                # print(f"[policy] observation timestamps: {anchor_timestamps}, action timestamps: {reply.get('timestamps', None) if reply else None}")
                if reply is None or "actions_tf" not in reply:
                    print("[policy] Inference timeout or malformed reply")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue

                actions_tf = reply.get("actions_tf")
                has_gripper_actions = bool(
                    actions_tf
                    and (
                        "gripper_left_gripper_width" in actions_tf
                        or "gripper_right_gripper_width" in actions_tf
                    )
                )
                action_timestamps = np.asarray(reply.get("timestamps", []), dtype=float)
                print(f"[policy] Received action chunk with keys: {list(actions_tf.keys())} and timestamps: {action_timestamps}")
                # print(f"[policy] received gripper widths: left={actions_tf.get('gripper_left_gripper_width', None)}, right={actions_tf.get('gripper_right_gripper_width', None)}")

                if args.plot_tracking and action_timestamps.size and anchor_timestamps is not None:
                    arm_cutoff = float(infer_end + max(args.executor_lookahead, args.arm_execution_latency, 0.0))
                    gripper_cutoff = float(infer_end + max(args.executor_lookahead, args.gripper_execution_latency, 0.0))
                    first_action_ready = arm_cutoff if not has_gripper_actions else max(arm_cutoff, gripper_cutoff)
                    obs_age = float(infer_end - float(np.asarray(anchor_timestamps).reshape(-1)[-1]))
                    latency_samples.append(
                        {
                            "obs_age_s": obs_age,
                            "infer_s": float(infer_end - infer_start),
                            "queue_delay_s": float(time.monotonic() - infer_start),
                            "first_action_delay_s": float(action_timestamps[0] - first_action_ready),
                        }
                    )

                if not actions_tf:
                    print("[policy] Missing actions in reply")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue

                # Drop actions that are already outdated when accounting for observation,
                # inference, and execution latency. The policy timestamps are anchored to
                # the observation stream, so any desired timestamp that lands before the
                # soonest achievable execution time would ask the robot to "time travel".
                # Following PD1.2, discard those actions instead of shifting the entire
                # chunk forward.
                if action_timestamps.size:
                    arm_cutoff = float(infer_end + max(args.executor_lookahead, args.arm_execution_latency, 0.0))
                    gripper_cutoff = float(infer_end + max(args.executor_lookahead, args.gripper_execution_latency, 0.0))
                    finite_ts = np.asarray(action_timestamps, dtype=float).reshape(-1)
                    valid_mask = np.isfinite(finite_ts) & (finite_ts > arm_cutoff)
                    if has_gripper_actions:
                        valid_mask &= finite_ts > gripper_cutoff
                    if not np.any(valid_mask):
                        cutoff_desc = f"arm>{arm_cutoff:.3f}s"
                        if has_gripper_actions:
                            cutoff_desc += f", gripper>{gripper_cutoff:.3f}s"
                        print(f"[policy] Dropping action chunk; all desired timestamps precede execution cutoffs ({cutoff_desc}).")
                        if stop_event.wait(timeout=control_dt):
                            break
                        continue

                    first_valid_idx = int(np.argmax(valid_mask))
                    if first_valid_idx > 0:
                        if has_gripper_actions:
                            msg = (
                                f"[policy] Skipping {first_valid_idx} stale actions to match "
                                f"arm/gripper execution latency (>{arm_cutoff:.3f}s/{gripper_cutoff:.3f}s)."
                            )
                        else:
                            msg = (
                                f"[policy] Skipping {first_valid_idx} stale actions to match "
                                f"arm execution latency (>{arm_cutoff:.3f}s)."
                            )
                        print(msg)
                        # Lookahead effect - substract the execution latency to keep the action timing consistent
                        action_timestamps -= max(args.executor_lookahead, args.arm_execution_latency, 0.0)
                        if args.plot_tracking:
                            # add all actions and timestamps, including the ones being skipped
                            scheduled_actions_raw = build_scheduled_actions(
                                actions_tf=actions_tf,
                                timestamps=action_timestamps,
                                fallback_dt=control_dt,
                                now=time.monotonic(),
                                gripper_width_offset=args.gripper_width_command_offset,
                            )
                            tracking_policy_raw.extend(
                                [(a.timestamp, a.payload) for a in scheduled_actions_raw]
                            )
                        action_timestamps = action_timestamps[first_valid_idx:]
                        actions_tf = {
                            key: (None if val is None else np.asarray(val)[first_valid_idx:])
                            for key, val in actions_tf.items()
                        }
  
                scheduled_actions = build_scheduled_actions(
                    actions_tf=actions_tf,
                    timestamps=action_timestamps,
                    fallback_dt=control_dt,
                    now=time.monotonic(),
                    gripper_width_offset=args.gripper_width_command_offset,
                )
                if scheduled_actions:
                    if args.debug_actions and debug_request_queue is not None:
                        current_obs = robot.get_latest_observation()
                        current_pose = None
                        if current_obs is not None:
                            current_pose = {
                                "gripper_left_tf": current_obs.left_tf.copy(),
                                "gripper_right_tf": current_obs.right_tf.copy(),
                                "head_tf": current_obs.head_tf.copy(),
                            }
                        # Plot scheduled actions against the robot pose expressed
                        # in the same (model) frame the controller uses.
                        request = DebugActionRequest(
                            actions=scheduled_actions,
                            current_pose=current_pose,
                            image_obs=debug_images,
                        )
                        debug_request_queue.put(request)
                        while not stop_event.is_set():
                            if request.decision_event.wait(timeout=0.1):
                                break
                        if not request.decision_event.is_set():
                            continue
                        if not request.approved:
                            print("[debug] Action batch rejected; skipping execution.")
                            continue

                    if (
                        debug_image_queue is not None
                        and debug_images is not None
                        and not debug_image_queue.full()
                    ):
                        debug_image_counter[0] += 1
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        output_dir = debug_media_root if debug_media_root is not None else Path(args.debug_video_dir)
                        batch_tag = f"{stamp}_{debug_image_counter[0]:03d}"
                        debug_image_queue.put(
                            DebugImageRequest(
                                image_obs=debug_images,
                                output_dir=output_dir,
                                batch_tag=batch_tag,
                                bgr_to_rgb=True,
                            )
                        )
                        if debug_env_obs_queue is not None and not debug_env_obs_queue.full():
                            debug_env_obs = {k: np.asarray(v).copy() for k, v in obs_dict.items()}
                            debug_env_obs_queue.put(
                                DebugEnvObsRequest(
                                    env_obs=debug_env_obs,
                                    output_dir=output_dir,
                                    batch_tag=batch_tag,
                                )
                            )
                        elif debug_env_obs_queue is not None:
                            print("[debug] Env obs queue full; dropping env obs batch.")
                    elif debug_image_queue is not None and debug_images is not None:
                        print("[debug] Image queue full; dropping debug image batch.")

                    if args.plot_tracking:
                        tracking_cmd.extend([(a.timestamp, a.payload) for a in scheduled_actions])
                    robot.queue_actions(scheduled_actions)
                    if args.plot_tracking:
                        obs = robot.get_latest_observation()
                        if obs is not None:
                            payload = {
                                "gripper_left_tf": obs.left_tf.copy(),
                                "gripper_right_tf": obs.right_tf.copy(),
                                "gripper_left_gripper_width": np.array([obs.left_width], dtype=float),
                                "gripper_right_gripper_width": np.array([obs.right_width], dtype=float),
                            }
                            backend = getattr(robot, "_backend", None)
                            if backend is not None and hasattr(backend, "get_latest_robot_state"):
                                snapshot = backend.get_latest_robot_state()
                                if snapshot is not None and getattr(snapshot, "is_valid", True) and hasattr(snapshot, "odom_SE2"):
                                    payload["odom_SE2"] = np.asarray(snapshot.odom_SE2, dtype=float).copy()
                            tracking_state.append((obs.timestamp, payload))

                elapsed = time.monotonic() - start
                wait_time = max(0.0, inference_period - elapsed)
                print(f"[policy] Inference cycle took {elapsed:.3f}s, waiting {wait_time:.3f}s until next cycle.")
                if stop_event.wait(timeout=wait_time):
                    break

        threads.append(threading.Thread(target=inference_worker, name="policy-inference", daemon=True))

        for thread in threads:
            thread.start()

        try:
            while True:
                if debug_request_queue is not None:
                    _handle_debug_requests(debug_request_queue)
                if not all(thread.is_alive() for thread in threads):
                    break
                if stop_event.wait(timeout=0.1):
                    break
        finally:
            if debug_request_queue is not None:
                _reject_pending_debug_requests(debug_request_queue)
    except KeyboardInterrupt:
        print("[main] Interrupted, shutting down...")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        if debug_image_stop is not None and debug_image_queue is not None:
            debug_image_stop.set()
            debug_image_queue.put(None)
        if debug_image_thread is not None:
            debug_image_thread.join(timeout=1.0)
        if debug_env_obs_stop is not None and debug_env_obs_queue is not None:
            debug_env_obs_stop.set()
            debug_env_obs_queue.put(None)
        if debug_env_obs_thread is not None:
            debug_env_obs_thread.join(timeout=1.0)
        if debug_video_thread is not None:
            debug_video_thread.join(timeout=1.0)
        if tracking_state_thread is not None:
            tracking_state_thread.join(timeout=1.0)
        if camera_streamer is not None:
            camera_streamer.stop()
        if policy_client is not None:
            policy_client.close()
        if args.plot_tracking and not args.sim_only:
            try:
                import matplotlib.pyplot as plt  # type: ignore
            except Exception as exc:  # pragma: no cover
                print(f"[plot] Unable to import matplotlib: {exc}")
            else:
                # Sample latest robot states to align with commands
                # plotting may have already captured many states via the sampler; append the freshest one as well
                obs = robot.get_latest_observation()
                if obs is not None:
                    payload = {
                        "gripper_left_tf": obs.left_tf.copy(),
                        "gripper_right_tf": obs.right_tf.copy(),
                        "gripper_left_gripper_width": np.array([obs.left_width], dtype=float),
                        "gripper_right_gripper_width": np.array([obs.right_width], dtype=float),
                    }
                    backend = getattr(robot, "_backend", None)
                    if backend is not None and hasattr(backend, "get_latest_robot_state"):
                        snapshot = backend.get_latest_robot_state()
                        if snapshot is not None and getattr(snapshot, "is_valid", True) and hasattr(snapshot, "odom_SE2"):
                            payload["odom_SE2"] = np.asarray(snapshot.odom_SE2, dtype=float).copy()
                    tracking_state.append((obs.timestamp, payload))

                def _tf_pos(tf: np.ndarray) -> np.ndarray:
                    return np.asarray(tf, dtype=float).reshape(4, 4)[:3, 3]

                def _series_to_lines(series: List[Tuple[float, Dict[str, np.ndarray]]], eff: str):
                    series_sorted = sorted(
                        series,
                        key=lambda item: float(np.asarray(item[0]).ravel()[0]) if item and item[0] is not None else 0.0,
                    )
                    ts: List[float] = []
                    xs: List[float] = []
                    ys: List[float] = []
                    zs: List[float] = []
                    for t, payload in series_sorted:
                        tf_keys = (
                            f"{eff}_tf",
                            f"gripper_{eff}_tf",
                            eff,
                        )
                        chosen = None
                        for tf_key in tf_keys:
                            if tf_key in payload:
                                chosen = payload[tf_key]
                                break
                        if chosen is None:
                            continue
                        pos = _tf_pos(chosen)
                        ts.append(t)
                        xs.append(pos[0])
                        ys.append(pos[1])
                        zs.append(pos[2])
                    return ts, xs, ys, zs

                def _series_to_width(series: List[Tuple[float, Dict[str, np.ndarray]]], eff: str):
                    series_sorted = sorted(
                        series,
                        key=lambda item: float(np.asarray(item[0]).ravel()[0]) if item and item[0] is not None else 0.0,
                    )
                    ts: List[float] = []
                    vals: List[float] = []
                    for t, payload in series_sorted:
                        width_keys = (
                            f"{eff}_gripper_width",
                            f"gripper_{eff}_gripper_width",
                        )
                        chosen = None
                        for w_key in width_keys:
                            if w_key in payload:
                                chosen = payload[w_key]
                                break
                        if chosen is None:
                            continue
                        width_val = float(np.asarray(chosen).reshape(-1)[0])
                        ts.append(t)
                        vals.append(width_val)
                    return ts, vals

                def _series_to_base_velocity(series: List[Tuple[float, Dict[str, np.ndarray]]]):
                    series_sorted = sorted(
                        series,
                        key=lambda item: float(np.asarray(item[0]).ravel()[0]) if item and item[0] is not None else 0.0,
                    )
                    ts: List[float] = []
                    xs: List[float] = []
                    ys: List[float] = []
                    yaws: List[float] = []
                    for t, payload in series_sorted:
                        if "odom_SE2" not in payload:
                            continue
                        mat = np.asarray(payload["odom_SE2"], dtype=float).reshape(3, 3)
                        ts.append(t)
                        xs.append(float(mat[0, 2]))
                        ys.append(float(mat[1, 2]))
                        yaws.append(float(np.arctan2(mat[1, 0], mat[0, 0])))
                    if len(ts) < 2:
                        return [], [], [], []
                    ts_arr = np.asarray(ts, dtype=float)
                    xs_arr = np.asarray(xs, dtype=float)
                    ys_arr = np.asarray(ys, dtype=float)
                    yaw_arr = np.unwrap(np.asarray(yaws, dtype=float))
                    dt = np.diff(ts_arr)
                    valid = dt > 1e-6
                    if not np.any(valid):
                        return [], [], [], []
                    vx = np.diff(xs_arr) / dt
                    vy = np.diff(ys_arr) / dt
                    wz = np.diff(yaw_arr) / dt
                    ts_mid = (ts_arr[:-1] + ts_arr[1:]) * 0.5
                    return list(ts_mid[valid]), list(vx[valid]), list(vy[valid]), list(wz[valid])

                plt.clf()
                fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex="row")
                eff_list = ["left", "right"]
                for col, eff in enumerate(eff_list):
                    ts_cmd, xs_cmd, ys_cmd, zs_cmd = _series_to_lines(tracking_cmd, eff)
                    ts_act, xs_act, ys_act, zs_act = _series_to_lines(tracking_state, eff)
                    ts_exec, xs_exec, ys_exec, zs_exec = _series_to_lines(tracking_exec, eff)
                    ts_policy_raw, xs_policy_raw, ys_policy_raw, zs_policy_raw = _series_to_lines(tracking_policy_raw, eff)
                    for row, (cmd, act, label) in enumerate(
                        zip((xs_cmd, ys_cmd, zs_cmd), (xs_act, ys_act, zs_act), ("x", "y", "z"))
                    ):
                        ax = axes[row][col]
                        # ax.plot(ts_cmd, cmd, linestyle="--", alpha=0.7, label=f"{eff} {label} cmd")
                        ax.scatter(ts_exec, {"x": xs_exec, "y": ys_exec, "z": zs_exec}[label],
                                    s=14, alpha=0.8, marker="+", color='m',
                                    label=f"{eff} {label} exec")
                        if ts_policy_raw:
                            ax.scatter(
                                ts_policy_raw,
                                {"x": xs_policy_raw, "y": ys_policy_raw, "z": zs_policy_raw}[label],
                                s=14,
                                alpha=0.6,
                                marker="o",
                                color="b",
                                label=f"{eff} {label} policy (raw)",
                            )
                        ax.scatter(ts_cmd, cmd, s=14, alpha=0.9, marker="o", color='r', label=f"{eff} {label} cmd")
                        ax.plot(ts_act, act, linestyle="-", alpha=0.9, label=f"{eff} {label} act")
                        ax.legend(loc="upper right")
                        ax.set_ylabel(label)
                        if row == 0:
                            ax.set_title(f"{eff} arm")
                    # gripper widths
                    ax_w = axes[-2][col]
                    ts_w_cmd, vals_w_cmd = _series_to_width(tracking_cmd, eff)
                    ts_w_act, vals_w_act = _series_to_width(tracking_state, eff)
                    ts_w_exec, vals_w_exec = _series_to_width(tracking_exec, eff)
                    ts_w_policy_raw, vals_w_policy_raw = _series_to_width(tracking_policy_raw, eff)
                    # ax_w.plot(ts_w_cmd, vals_w_cmd, linestyle="--", alpha=0.7, label=f"{eff} width cmd")
                    ax_w.scatter(ts_w_cmd, vals_w_cmd, s=14, alpha=0.7, marker="x", color='r', label=f"{eff} width cmd")
                    if ts_w_exec:
                        ax_w.scatter(ts_w_exec, vals_w_exec, s=14, alpha=0.8, marker="o", color='m', label=f"{eff} width exec")
                    if ts_w_policy_raw:
                        ax_w.scatter(
                            ts_w_policy_raw,
                            vals_w_policy_raw,
                            s=14,
                            alpha=0.6,
                            marker="o",
                            color="b",
                            label=f"{eff} width policy (raw)",
                        )
                    # ax_w.plot(ts_w_act, vals_w_act, linestyle="-", alpha=0.9, label=f"{eff} width act")
                    ax_w.scatter(ts_w_act, vals_w_act, s=14, alpha=0.9, marker=".", color='g', label=f"{eff} width act")
                    ax_w.set_ylabel("width")
                    ax_w.legend(loc="upper right")
                ts_base, vx_base, vy_base, wz_base = _series_to_base_velocity(tracking_state)
                ax_base = axes[-1][0]
                if ts_base:
                    ax_base.plot(ts_base, vx_base, label="vx")
                    ax_base.plot(ts_base, vy_base, label="vy")
                    ax_base.plot(ts_base, wz_base, label="wz")
                    ax_base.legend(loc="upper right")
                ax_base.set_ylabel("base vel")
                ax_base.set_title("base")
                axes[-1][1].axis("off")
                axes[-2][1].set_xlabel("time (s)")
                axes[-1][0].set_xlabel("time (s)")
                plt.tight_layout()
                if latency_samples:
                    obs_age_vals = [s["obs_age_s"] for s in latency_samples]
                    infer_vals = [s["infer_s"] for s in latency_samples]
                    first_action_delay = [s["first_action_delay_s"] for s in latency_samples if np.isfinite(s["first_action_delay_s"])]
                    print(
                        "[latency] obs_age_s med/p95: "
                        f"{np.median(obs_age_vals):.4f}/{np.percentile(obs_age_vals, 95):.4f}, "
                        f"infer_s med/p95: {np.median(infer_vals):.4f}/{np.percentile(infer_vals, 95):.4f}, "
                        f"first_action_delay_s med/p95: "
                        f"{(np.median(first_action_delay) if first_action_delay else float('nan')):.4f}/"
                        f"{(np.percentile(first_action_delay, 95) if first_action_delay else float('nan')):.4f}"
                    )
                out_path = Path(PROJECT_ROOT) / "tracking_plot.png"
                try:
                    fig.savefig(out_path)
                    print(f"[plot] Saved tracking plot to {out_path}")
                except Exception as exc:  # pragma: no cover
                    print(f"[plot] Failed to save plot: {exc}")
                plt.close(fig)
        robot.stop()


if __name__ == "__main__":
    main()
