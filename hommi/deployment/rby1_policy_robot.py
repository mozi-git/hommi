"""High-level robot interface tailored for policy streaming.

This module exposes :class:`RBY1PolicyRobot`, a thin convenience wrapper around
the existing :class:`~control.rby1_wbc.RBY1WBC` whole-body controller.  The
class keeps a short history of end-effector poses expressed in Cartesian space
and provides a helper for dispatching Cartesian targets produced by a learned
policy.  The implementation intentionally mirrors the behaviour in
``scripts/rby1_wbc_traj.py`` where differential IK solutions are sent to the
pybind realtime driver.

Typical usage::

    robot = RBY1PolicyRobot()
    robot.start()
    robot.wait_until_ready()
    obs = robot.get_observation_window(horizon=2)
    robot.apply_action(policy_action)

The class is thread-safe and designed to run alongside a camera streamer in a
policy inference loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Sequence

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import yaml

HOMMI_ROOT = Path(__file__).resolve().parents[2]
RBY1_WBC_ROOT = HOMMI_ROOT / "deps" / "rby1-wbc"
if RBY1_WBC_ROOT.exists():
    import sys

    rby1_wbc_path = str(RBY1_WBC_ROOT)
    if rby1_wbc_path not in sys.path:
        sys.path.insert(0, rby1_wbc_path)

from rby1.frame_transforms import MODEL_TO_TCP_FRAME, TCP_TO_MODEL_FRAME
from rby1.whole_body_ik import RBY1WholeBodyIK
from control.rby1_wbc import RBY1WBC

try:  # pragma: no cover - optional dependency
    import mujoco.viewer as mj_viewer
except (ImportError, AttributeError):
    mj_viewer = None


@dataclass(frozen=True)
class RobotObservation:
    """Compact representation of a single policy observation."""

    timestamp: float
    left_tf: np.ndarray
    right_tf: np.ndarray
    head_tf: np.ndarray
    left_width: float
    right_width: float


@dataclass(frozen=True)
class ScheduledAction:
    """Action queued for execution at a wall-clock timestamp."""

    timestamp: float
    duration: float
    payload: Dict[str, np.ndarray]


@dataclass(frozen=True)
class _SimSnapshot:
    qpos: np.ndarray
    is_valid: bool = True

def _build_interp1d(t: np.ndarray, values: np.ndarray) -> interp1d:
    return interp1d(
        t,
        values,
        axis=0,
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )


class _PoseInterpolator:
    def __init__(self, timestamps: np.ndarray, poses: np.ndarray) -> None:
        self._t = np.asarray(timestamps, dtype=float)
        self._t_min = float(self._t[0])
        self._t_max = float(self._t[-1])
        self._pos_interp = _build_interp1d(self._t, poses[:, :3])
        if len(self._t) >= 2:
            self._rot_slerp = Slerp(self._t, Rotation.from_rotvec(poses[:, 3:]))
            self._rot_static = None
        else:
            self._rot_slerp = None
            self._rot_static = Rotation.from_rotvec(poses[0, 3:])

    def __call__(self, query_t: np.ndarray) -> np.ndarray:
        query = np.asarray(query_t, dtype=float)
        needs_expand = query.ndim == 0
        if needs_expand:
            query = query[None]
        query = np.clip(query, self._t_min, self._t_max)
        pos = self._pos_interp(query)
        if self._rot_slerp is not None:
            rot = self._rot_slerp(query).as_rotvec()
        else:
            rot = np.tile(self._rot_static.as_rotvec(), (len(query), 1))
        pose = np.concatenate([pos, rot], axis=-1)
        if needs_expand:
            pose = pose[0]
        return pose

def _tf_to_posevec(tf: np.ndarray) -> np.ndarray:
    pose = np.zeros(6, dtype=float)
    pose[:3] = tf[:3, 3]
    pose[3:] = Rotation.from_matrix(tf[:3, :3]).as_rotvec()
    return pose


def _posevec_to_tf(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    tf = np.tile(np.eye(4), (pose.shape[0], 1, 1))
    tf[:, :3, 3] = pose[:, :3]
    tf[:, :3, :3] = Rotation.from_rotvec(pose[:, 3:]).as_matrix()
    return tf

class _SimBackend:
    """Minimal drop-in replacement for :class:`RBY1WBC` used for debugging."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sim_frequency: float = 20.0,
        enable_viewer: bool = False,
    ) -> None:
        project_root = RBY1_WBC_ROOT
        default_model = project_root / "model" / "rby1" / "rby1.xml"
        default_config = project_root / "config" / "wbc.yaml"
        self.model_path = str(model_path or default_model)
        self.init_config_path = default_config
        try:
            with Path(self.init_config_path).open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception:
            config = {}
        self._init_position = config.get("init_position", {})
        self.trajectory_frequency_hz = sim_frequency
        self._model = mujoco.MjModel.from_xml_path(self.model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._ik = RBY1WholeBodyIK()
        self._state_lock = threading.Lock()
        self._apply_init_position()
        self._snapshot = _SimSnapshot(self._data.qpos.copy())
        self._stop = threading.Event()
        self._viewer_enabled = enable_viewer and (mj_viewer is not None)
        self._viewer_thread: Optional[threading.Thread] = None
        self._viewer: Optional[Any] = None
        self._viewer_data: Optional[mujoco.MjData] = None
        self._gripper_widths = (0.08, 0.08)
        self._apply_init_position()

    def start(self) -> None:
        if not self._viewer_enabled or self._viewer_thread is not None:
            return
        self._stop.clear()
        self._viewer_thread = threading.Thread(target=self._viewer_loop, name="sim-viewer", daemon=True)
        self._viewer_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
            self._viewer_thread = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._viewer = None
        self._viewer_data = None

    def wait_for_first_state(self, timeout_sec: float = 0.0) -> _SimSnapshot:
        return self._snapshot

    def get_latest_robot_state(self) -> _SimSnapshot:
        with self._state_lock:
            return _SimSnapshot(self._data.qpos.copy())

    def snapshot_to_qpos(self, snapshot: _SimSnapshot) -> np.ndarray:
        return snapshot.qpos.copy()

    def get_latest_gripper_widths(self) -> tuple[float, float]:
        return self._gripper_widths

    def update_targets(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        head_lookatpoint: Optional[np.ndarray] = None,
        duration: float = 0.1,
        timestamp: Optional[float] = None,
    ) -> None:
        sol_qpos, sol_vel, success, _ = self._ik.solve(
            left_target_pos=left_pos,
            left_target_quat=left_quat,
            right_target_pos=right_pos,
            right_target_quat=right_quat,
            head_target_pos=head_pos,
            head_target_quat=head_quat,
            current_qpos=self._data.qpos.copy(),
            dt=max(duration, 1e-3),
        )
        if not success:
            print("[sim] IK failed to reach target")  # pragma: no cover - debug aid
            return
        with self._state_lock:
            self._data.qpos[:] = sol_qpos
            self._data.qvel[:] = sol_vel
            mujoco.mj_forward(self._model, self._data)
            self._snapshot = _SimSnapshot(sol_qpos.copy())
            lw, rw = self._gripper_widths
            if left_width is not None:
                lw = float(left_width)
            if right_width is not None:
                rw = float(right_width)
            self._gripper_widths = (lw, rw)
    def _apply_init_position(self) -> None:
        init = self._init_position
        if not init:
            return
        torso = np.asarray(init.get("torso", []), dtype=float) if init.get("torso") else None
        left = np.asarray(init.get("left_arm", []), dtype=float) if init.get("left_arm") else None
        right = np.asarray(init.get("right_arm", []), dtype=float) if init.get("right_arm") else None
        head = np.asarray(init.get("head", []), dtype=float) if init.get("head") else None
        if all(item is None for item in (torso, left, right, head)):
            return
        with self._state_lock:
            qpos = self._data.qpos.copy()
            if torso is not None:
                for idx, value in zip(self._ik.torso_qpos_indices, torso):
                    qpos[int(idx)] = float(value)
            if left is not None:
                for idx, value in zip(self._ik.left_arm_qpos_indices, left):
                    qpos[int(idx)] = float(value)
            if right is not None:
                for idx, value in zip(self._ik.right_arm_qpos_indices, right):
                    qpos[int(idx)] = float(value)
            if head is not None:
                for idx, value in zip(self._ik.head_qpos_indices, head):
                    qpos[int(idx)] = float(value)
            self._data.qpos[:] = qpos
            mujoco.mj_forward(self._model, self._data)

    def _viewer_loop(self) -> None:
        if mj_viewer is None:
            return
        try:
            self._viewer_data = mujoco.MjData(self._model)
            self._viewer = mj_viewer.launch_passive(
                model=self._model, data=self._viewer_data, show_left_ui=False, show_right_ui=False
            )
            mujoco.mjv_defaultFreeCamera(self._model, self._viewer.cam)
        except Exception as exc:  # pragma: no cover - optional viewer
            print(f"[sim] Failed to start viewer: {exc}")
            return
        while not self._stop.is_set():
            if self._viewer_data is None or self._viewer is None:
                break
            with self._state_lock:
                self._viewer_data.qpos[:] = self._data.qpos
                self._viewer_data.qvel[:] = self._data.qvel
                mujoco.mj_forward(self._model, self._viewer_data)
            self._viewer.sync()
            time.sleep(1.0 / 60.0)

class RBY1PolicyRobot:
    """High-level helper for streaming policy commands to the robot.

    The object internally owns a :class:`RBY1WBC` instance and mirrors the
    MuJoCo model so that policy-facing Cartesian states can be produced.  A
    fixed-size deque keeps the most recent observations which are returned in a
    format that matches the expected keys of the policy server (e.g.
    ``gripper_left_eef_pos``).
    """

    LEFT_SITE = "end_effector_l"
    RIGHT_SITE = "end_effector_r"
    HEAD_SITE = "head"

    def __init__(
        self,
        config_path: str | None = None,
        buffer_size: int = 64,
        use_sim: bool = False,
        sim_model_path: Optional[str] = None,
        sim_viewer: bool = False,
        max_pos_speed: float = 0.25,    # m/s
        max_rot_speed: float = 0.2,    # rad/s
        max_gripper_speed: float = 0.1,
        gripper_min_width: float = 0.0,
        gripper_max_width: float = 0.085,
    ) -> None:
        if use_sim:
            self._backend: Any = _SimBackend(model_path=sim_model_path, enable_viewer=sim_viewer)
        else:
            self._backend = RBY1WBC(config_path=config_path) if config_path else RBY1WBC()
        self._buffer: Deque[RobotObservation] = deque(maxlen=int(max(buffer_size, 1)))
        self._buffer_lock = threading.Lock()
        self._buffer_ready = threading.Event()
        self._stop_event = threading.Event()
        self._buffer_thread: Optional[threading.Thread] = None
        self._command_thread: Optional[threading.Thread] = None
        self._command_stop = threading.Event()
        self._action_lock = threading.Lock()
        self._action_buffer: Deque[ScheduledAction] = deque()
        self._last_executed_action: Optional[ScheduledAction] = None
        self._execution_hook: Optional[Callable[[float, Dict[str, np.ndarray]], None]] = None
        self._max_pos_speed = float(max_pos_speed)
        self._max_rot_speed = float(max_rot_speed)
        self._max_gripper_speed = float(max_gripper_speed)
        self._gripper_limits = (float(gripper_min_width), float(gripper_max_width))

        # Mirror of the MuJoCo model used for FK computations in the streaming
        # thread.  The worker keeps an internal model already, but sharing it
        # across threads is unsafe, so we create our own lightweight copy.
        self._model = mujoco.MjModel.from_xml_path(self._backend.model_path)
        self._data = mujoco.MjData(self._model)
        self._left_site = self._model.site(self.LEFT_SITE).id
        self._right_site = self._model.site(self.RIGHT_SITE).id
        self._head_site = self._model.site(self.HEAD_SITE).id

    @property
    def dt(self) -> float:
        """Return the IK loop period of the underlying controller."""

        return 1.0 / float(self._backend.trajectory_frequency_hz)

    def set_execution_hook(self, hook: Optional[Callable[[float, Dict[str, np.ndarray]], None]]) -> None:
        """Register a callback invoked after each command is sent to the backend."""

        self._execution_hook = hook

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the realtime controller threads."""

        self._backend.start()
        if self._buffer_thread is not None and self._buffer_thread.is_alive():
            return
        self._stop_event.clear()
        self._buffer_thread = threading.Thread(
            target=self._buffer_worker,
            name="policy-robot-buffer",
            daemon=True,
        )
        self._buffer_thread.start()
        self._start_command_thread()

    def stop(self) -> None:
        """Stop controller threads and flush buffers."""

        self._stop_event.set()
        self._command_stop.set()
        if self._buffer_thread is not None:
            self._buffer_thread.join(timeout=1.0)
            self._buffer_thread = None
        if self._command_thread is not None:
            self._command_thread.join(timeout=1.0)
            self._command_thread = None
        self._backend.stop()
        with self._buffer_lock:
            self._buffer.clear()
        self._buffer_ready.clear()
        self._command_stop.clear()
        with self._action_lock:
            self._action_buffer.clear()
            self._last_executed_action = None

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        """Block until the first valid snapshot is available."""

        deadline = time.monotonic() + max(timeout, 0.0)
        self._backend.wait_for_first_state(timeout_sec=timeout)
        remaining = max(0.0, deadline - time.monotonic())
        if not self._buffer_ready.wait(timeout=remaining):
            raise TimeoutError("Timed out waiting for buffered robot observations")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _buffer_worker(self) -> None:
        """Continuously fetch robot snapshots and append them to the buffer."""

        poll_period = max(self.dt, 0.01)
        while not self._stop_event.is_set():
            snapshot = self._backend.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                qpos = self._backend.snapshot_to_qpos(snapshot)
                if qpos is not None:
                    observation = self._qpos_to_observation(qpos, time.monotonic())
                    with self._buffer_lock:
                        self._buffer.append(observation)
                        self._buffer_ready.set()
            time.sleep(poll_period)

    def _start_command_thread(self) -> None:
        if self._command_thread is not None and self._command_thread.is_alive():
            return
        self._command_stop.clear()
        self._command_thread = threading.Thread(
            target=self._command_worker,
            name="policy-robot-command",
            daemon=True,
        )
        self._command_thread.start()

    def _command_worker(self) -> None:
        command_dt = 1.0 / float(self._backend.trajectory_frequency_hz)
        while not self._command_stop.is_set():
            now = time.monotonic()
            query_time = now
            payload = self._sample_scheduled_payload(query_time)
            if payload is not None and {"gripper_left_tf", "gripper_right_tf"}.issubset(payload):
                self.apply_action(payload, duration=command_dt, timestamp=now)
                with self._action_lock:
                    # Store the last command actually sent to the robot for smoother interpolation.
                    payload_copy = {k: np.asarray(v, dtype=float).copy() if isinstance(v, np.ndarray) else v for k, v in payload.items()}
                    self._last_executed_action = ScheduledAction(timestamp=query_time, duration=command_dt, payload=payload_copy)
                if self._execution_hook is not None:
                    try:
                        self._execution_hook(query_time, payload_copy)
                    except Exception:
                        pass
            wait_time = max(0.0, command_dt - (time.monotonic() - now))
            if self._command_stop.wait(timeout=wait_time):
                break

    def wait_for_observations(self, count: int, timeout: float = 2.0) -> bool:
        """Block until the buffer stores ``count`` observations.

        Returns ``True`` when the desired number of entries are available and
        ``False`` if the timeout expires first.
        """

        count = int(max(count, 1))
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() <= deadline:
            with self._buffer_lock:
                if len(self._buffer) >= count:
                    return True
            time.sleep(min(self.dt, 0.01))
        return False

    def _qpos_to_observation(self, qpos: np.ndarray, timestamp: float) -> RobotObservation:
        """Convert a joint configuration into Cartesian observations."""

        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)

        def _site_tf(site_id: int) -> np.ndarray:
            tf = np.eye(4, dtype=float)
            tf[:3, 3] = self._data.site_xpos[site_id].astype(float)
            tf[:3, :3] = self._data.site_xmat[site_id].reshape(3, 3).astype(float)
            return tf

        left_tf = _site_tf(self._left_site)
        right_tf = _site_tf(self._right_site)
        head_tf = _site_tf(self._head_site)

        left_width, right_width = self._backend.get_latest_gripper_widths()

        return RobotObservation(
            timestamp=float(timestamp),
            left_tf=left_tf,
            right_tf=right_tf,
            head_tf=head_tf,
            left_width=float(left_width),
            right_width=float(right_width),
        )

    def get_latest_observation(self) -> Optional[RobotObservation]:
        with self._buffer_lock:
            return self._buffer[-1] if self._buffer else None

    def sample_observations_at(self, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Interpolate robot observations at arbitrary timestamps."""

        query_ts = np.asarray(timestamps, dtype=float).reshape(-1)
        if query_ts.size == 0:
            raise ValueError("timestamps must contain at least one entry")

        with self._buffer_lock:
            if not self._buffer:
                raise RuntimeError("Robot observation buffer is empty")
            buffer_list = list(self._buffer)

        buffer_timestamps = np.array([obs.timestamp for obs in buffer_list], dtype=float)

        def _build_tf_interp(get_tf: Callable[[RobotObservation], np.ndarray]) -> _PoseInterpolator:
            poses = np.stack([_tf_to_posevec(get_tf(obs)) for obs in buffer_list], axis=0)
            return _PoseInterpolator(buffer_timestamps, poses)

        left_interp = _build_tf_interp(lambda obs: obs.left_tf)
        right_interp = _build_tf_interp(lambda obs: obs.right_tf)
        head_interp = _build_tf_interp(lambda obs: obs.head_tf)

        def _interp_scalar(attr: str) -> np.ndarray:
            values = np.stack([getattr(obs, attr) for obs in buffer_list], axis=0)
            return _build_interp1d(buffer_timestamps, values.reshape(-1, 1))

        left_width_interp = _interp_scalar("left_width")
        right_width_interp = _interp_scalar("right_width")

        left_tf = _posevec_to_tf(left_interp(query_ts))
        right_tf = _posevec_to_tf(right_interp(query_ts))
        head_tf = _posevec_to_tf(head_interp(query_ts))
        left_width = left_width_interp(query_ts)
        right_width = right_width_interp(query_ts)

        return {
            "timestamp": query_ts,
            "gripper_left_tf": left_tf,
            "gripper_right_tf": right_tf,
            "head_tf": head_tf,
            "gripper_left_gripper_width": left_width.reshape(-1, 1),
            "gripper_right_gripper_width": right_width.reshape(-1, 1),
        }

    def get_observation_window(
        self,
        horizon: int,
        step: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """Return stacked arrays compatible with the policy server.

        Parameters
        ----------
        horizon:
            Number of historical steps to collect (oldest to newest).
        step:
            Time interval between observations in seconds.
        """
        horizon = int(max(horizon, 1))
        with self._buffer_lock:
            if not self._buffer:
                raise RuntimeError("Robot observation buffer is empty")
            buffer_list = list(self._buffer)
            latest_timestamp = buffer_list[-1].timestamp
            desired_timestamps = latest_timestamp - step * np.arange(horizon - 1, -1, -1, dtype=float)
            buffer_timestamps = np.array([obs.timestamp for obs in buffer_list], dtype=float)

            def _build_tf_interp(get_tf: Callable[[RobotObservation], np.ndarray]) -> _PoseInterpolator:
                poses = np.stack([_tf_to_posevec(get_tf(obs)) for obs in buffer_list], axis=0)
                return _PoseInterpolator(buffer_timestamps, poses)

            left_interp = _build_tf_interp(lambda obs: obs.left_tf)
            right_interp = _build_tf_interp(lambda obs: obs.right_tf)
            head_interp = _build_tf_interp(lambda obs: obs.head_tf)

            def _interp_scalar(attr: str) -> np.ndarray:
                values = np.stack([getattr(obs, attr) for obs in buffer_list], axis=0)
                return _build_interp1d(buffer_timestamps, values.reshape(-1, 1))

            left_width_interp = _interp_scalar("left_width")
            right_width_interp = _interp_scalar("right_width")

            left_tf = _posevec_to_tf(left_interp(desired_timestamps))
            right_tf = _posevec_to_tf(right_interp(desired_timestamps))
            head_tf = _posevec_to_tf(head_interp(desired_timestamps))
            left_width = left_width_interp(desired_timestamps)
            right_width = right_width_interp(desired_timestamps)

            return {
                "timestamp": desired_timestamps,
                "gripper_left_tf": left_tf,
                "gripper_right_tf": right_tf,
                "head_tf": head_tf,
                "gripper_left_gripper_width": left_width.reshape(-1, 1),
                "gripper_right_gripper_width": right_width.reshape(-1, 1),
            }

        if len(observations) < horizon:
            observations = [observations[0]] * (horizon - len(observations)) + observations

        def stack_tf(attr: str) -> np.ndarray:
            return np.stack([getattr(obs, attr) for obs in observations], axis=0)

        def stack_scalar(attr: str) -> np.ndarray:
            return np.stack([getattr(obs, attr) for obs in observations], axis=0).reshape(-1, 1)

        timestamps = np.array([obs.timestamp for obs in observations], dtype=np.float64)
        return {
            "timestamp": timestamps,
            "gripper_left_tf": stack_tf("gripper_left_tf"),
            "gripper_right_tf": stack_tf("gripper_right_tf"),
            "head_tf": stack_tf("head_tf"),
            "gripper_left_gripper_width": stack_scalar("gripper_left_gripper_width"),
            "gripper_right_gripper_width": stack_scalar("gripper_right_gripper_width"),
        }

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def _clamp_gripper_width(self, value: float) -> float:
        return float(np.clip(value, self._gripper_limits[0], self._gripper_limits[1]))

    def _interpolate_tf(self, tf_a: np.ndarray, tf_b: np.ndarray, alpha: float) -> np.ndarray:
        tf_a = np.asarray(tf_a, dtype=float).reshape(4, 4)
        tf_b = np.asarray(tf_b, dtype=float).reshape(4, 4)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return tf_a
        if alpha >= 1.0:
            return tf_b
        pos = (1.0 - alpha) * tf_a[:3, 3] + alpha * tf_b[:3, 3]
        rot_interp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([tf_a[:3, :3], tf_b[:3, :3]], axis=0)))
        rot = rot_interp([alpha])[0].as_matrix()
        tf = np.eye(4, dtype=float)
        tf[:3, 3] = pos
        tf[:3, :3] = rot
        return tf

    def _interpolate_actions(
        self, previous: ScheduledAction, future: ScheduledAction, query_time: float
    ) -> Dict[str, np.ndarray]:
        if future.timestamp <= previous.timestamp:
            return dict(future.payload)
        alpha = np.clip((query_time - previous.timestamp) / (future.timestamp - previous.timestamp), 0.0, 1.0)
        payload: Dict[str, np.ndarray] = {}
        keys = set(previous.payload) | set(future.payload)
        for key in keys:
            a_val = previous.payload.get(key)
            b_val = future.payload.get(key)
            if a_val is None:
                payload[key] = b_val  # type: ignore[assignment]
                continue
            if b_val is None:
                payload[key] = a_val  # type: ignore[assignment]
                continue
            if key.endswith("_tf"):
                payload[key] = self._interpolate_tf(a_val, b_val, alpha)
            else:
                a_arr = np.asarray(a_val, dtype=float)
                b_arr = np.asarray(b_val, dtype=float)
                payload[key] = (1.0 - alpha) * a_arr + alpha * b_arr
        return payload

    def _sample_scheduled_payload(self, query_time: float) -> Optional[Dict[str, np.ndarray]]:
        def _contains_tf(payload: Dict[str, np.ndarray]) -> bool:
            return any(k.endswith("_tf") or k == "camera_head_lookatpoint" for k in payload)

        def _contains_gripper(payload: Dict[str, np.ndarray]) -> bool:
            return any("gripper_width" in k for k in payload)

        def _select_actions(predicate: Callable[[Dict[str, np.ndarray]], bool]) -> tuple[Optional[ScheduledAction], Optional[ScheduledAction]]:
            previous: Optional[ScheduledAction] = None
            future: Optional[ScheduledAction] = None
            for action in self._action_buffer:
                if not predicate(action.payload):
                    continue
                if action.timestamp > query_time:
                    future = action
                    break
                previous = action
            if (
                self._last_executed_action
                and self._last_executed_action.timestamp <= query_time
                and predicate(self._last_executed_action.payload)
            ):
                previous = self._last_executed_action
            return previous, future

        def _resolve_payload(previous: Optional[ScheduledAction], future: Optional[ScheduledAction], key_filter: Callable[[str], bool]) -> Optional[Dict[str, np.ndarray]]:
            if previous is None and future is None:
                return None
            if previous is None:
                payload = dict(future.payload)
            elif future is None:
                payload = dict(previous.payload)
            else:
                payload = self._interpolate_actions(previous, future, query_time)
            return {k: v for k, v in payload.items() if key_filter(k)}

        with self._action_lock:
            eef_prev, eef_future = _select_actions(_contains_tf)
            gripper_prev, gripper_future = _select_actions(_contains_gripper)

        eef_payload = _resolve_payload(eef_prev, eef_future, lambda k: k.endswith("_tf") or k == "camera_head_lookatpoint")
        gripper_payload = _resolve_payload(gripper_prev, gripper_future, lambda k: "gripper_width" in k)

        if eef_payload is None and gripper_payload is None:
            return None

        result: Dict[str, np.ndarray] = {}
        if eef_payload is not None:
            result.update(eef_payload)
        if gripper_payload is not None:
            result.update(gripper_payload)

        for key in ("gripper_left_gripper_width", "gripper_right_gripper_width"):
            if key in result:
                result[key] = self._clamp_gripper_width(float(np.asarray(result[key]).reshape(-1)[0]))
            # print(f"gripper command {key}: {result[key]}")  # pragma: no cover - debug aid
        return result

    def queue_actions(self, actions: Sequence[ScheduledAction]) -> None:
        """Store timestamped actions, letting the newest plan overwrite future entries."""
        now = time.monotonic()
        # filtered = [action for action in actions if action.timestamp > now]
        # if not filtered:
        #     return
        # print(f"action buffer timestamps: {[action.timestamp for action in self._action_buffer]}")  # pragma: no cover - debug aid
        with self._action_lock:
            while self._action_buffer and self._action_buffer[-1].timestamp >= actions[0].timestamp:
                self._action_buffer.pop()
            # Keep buffer compact by trimming old entries.
            while self._action_buffer and self._action_buffer[0].timestamp < now:
                self._action_buffer.popleft()
            self._action_buffer.extend(actions)
        # print(f"action buffer timestamps after queue: {[action.timestamp for action in self._action_buffer]}")  # pragma: no cover - debug aid

    def schedule_waypoint(self, payload: Dict[str, np.ndarray], timestamp: float, duration: float) -> None:
        """Compatibility wrapper for legacy callers."""

        self.queue_actions([ScheduledAction(timestamp=float(timestamp), duration=float(duration), payload=payload)])

    def _current_head_tf_model(self) -> np.ndarray:
        tf = np.eye(4, dtype=float)
        data = getattr(self._backend, "_data", None)
        if data is None:
            data = getattr(self._backend, "data", None)
        if data is None:
            raise RuntimeError("Backend does not expose MuJoCo data for head pose")
        tf[:3, 3] = np.asarray(data.site_xpos[self._head_site], dtype=float)
        tf[:3, :3] = np.asarray(data.site_xmat[self._head_site], dtype=float).reshape(3, 3)
        return tf

    def apply_action(
        self,
        action: Dict[str, np.ndarray],
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Forward a single Cartesian action to the WBC."""

        for key in ("gripper_left_tf", "gripper_right_tf"):
            if key not in action:
                raise KeyError(f"Missing required action field '{key}'")

        left_tf = np.asarray(action["gripper_left_tf"], dtype=float).reshape(4, 4)
        right_tf = np.asarray(action["gripper_right_tf"], dtype=float).reshape(4, 4)
        head_tf = action.get("head_tf")
        head_tf = None if head_tf is None else np.asarray(head_tf, dtype=float).reshape(4, 4)

        def _tf_to_pos_quat(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pos = tf[:3, 3]
            quat_xyzw = Rotation.from_matrix(tf[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            return pos, quat_wxyz

        left_pos, left_quat = _tf_to_pos_quat(left_tf)
        right_pos, right_quat = _tf_to_pos_quat(right_tf)
        head_pos = None
        head_quat = None
        head_model_tf = None
        if head_tf is not None:
            head_pos, head_quat = _tf_to_pos_quat(head_tf)
        head_lookatpoint = action.get("camera_head_lookatpoint")
        if head_lookatpoint is not None and self._backend._head_control_mode == "ik":
            assert head_tf is None, "Cannot specify both head_tf and camera_head_lookatpoint"
            head_lookatpoint = np.asarray(head_lookatpoint, dtype=float).reshape(3)
            head_model_tf = self._current_head_tf_model()
            head_tcp_tf = head_model_tf @ MODEL_TO_TCP_FRAME["head"]
            head_pos = head_model_tf[:3, 3]

            direction = head_lookatpoint - head_tcp_tf[:3, 3]
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-8:
                direction = head_tcp_tf[:3, 2]
                direction_norm = float(np.linalg.norm(direction))
            z_axis = direction / (direction_norm + 1e-8)

            current_tcp_x = head_tcp_tf[:3, :3][:, 0]
            x_axis = current_tcp_x - np.dot(current_tcp_x, z_axis) * z_axis
            x_norm = float(np.linalg.norm(x_axis))
            if x_norm < 1e-6:
                world_up = np.array([0.0, 0.0, 1.0], dtype=float)
                x_axis = world_up - np.dot(world_up, z_axis) * z_axis
                x_norm = float(np.linalg.norm(x_axis))
                if x_norm < 1e-6:
                    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    x_axis /= x_norm
            else:
                x_axis /= x_norm
            y_axis = np.cross(z_axis, x_axis)
            rot_tcp = np.stack([x_axis, y_axis, z_axis], axis=-1)
            rot_model = rot_tcp @ TCP_TO_MODEL_FRAME["head"][:3, :3]
            head_quat_xyzw = Rotation.from_matrix(rot_model).as_quat()
            head_quat = np.array([head_quat_xyzw[3], head_quat_xyzw[0], head_quat_xyzw[1], head_quat_xyzw[2]])

        left_width = float(action["gripper_left_gripper_width"]) if "gripper_left_gripper_width" in action else None
        right_width = float(action["gripper_right_gripper_width"]) if "gripper_right_gripper_width" in action else None

        # print(f"[rby1_policy] Applying action: left_pos={left_pos}, left_quat={left_quat}, right_pos={right_pos}, right_quat={right_quat}, left_width={left_width}, right_width={right_width}, head_pos={head_pos}, head_quat={head_quat}, head_lookatpoint={head_lookatpoint}, duration={duration}, timestamp={timestamp}")  # pragma: no cover - debug aid
        self._backend.update_targets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=head_pos,
            head_quat=head_quat,
            head_lookatpoint=head_lookatpoint,
            duration=self.dt if duration is None else float(duration),
            timestamp=timestamp,
        )


__all__ = ["RBY1PolicyRobot", "RobotObservation", "ScheduledAction"]
