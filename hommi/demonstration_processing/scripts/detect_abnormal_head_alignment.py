import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from omegaconf import OmegaConf

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    this_file = Path(__file__).resolve()
    added = False
    for parent in this_file.parents:
        candidate = parent / "deps" / "universal_manipulation_interface"
        if candidate.exists():
            import sys

            sys.path.append(str(candidate))
            added = True
            break
    if not added:
        raise
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

from hommi.common.replay_buffer import ReplayBuffer
from umi.common.pose_util import pose_to_mat

try:
    from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidate = parent / "deps" / "universal_manipulation_interface"
        if candidate.exists():
            sys.path.append(str(candidate))
            break
    from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep

register_codecs()


def _load_task_config(task_path: str):
    task_cfg = OmegaConf.load(task_path)
    wrapped = OmegaConf.create({})
    wrapped["task"] = task_cfg
    for key in task_cfg.keys():
        wrapped[key] = task_cfg[key]
    OmegaConf.resolve(wrapped)
    return wrapped.task


def _open_store(path: str):
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return zarr.DirectoryStore(path)
    if path.endswith(".zip"):
        return zarr.ZipStore(path, mode="r")
    if path.endswith(".zarr"):
        return zarr.DirectoryStore(path)
    raise ValueError(f"Unsupported replay buffer path: {path}")


def _decode_episode_names(names: Iterable) -> List[str]:
    decoded = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8"))
        else:
            decoded.append(str(name))
    return decoded


def _resolve_pose_key(
    replay_buffer: ReplayBuffer,
    override_key: Optional[str],
    prefix: str,
    side: str,
) -> str:
    if override_key:
        if override_key not in replay_buffer.data:
            raise KeyError(f"Pose key '{override_key}' not found in replay buffer.")
        return override_key
    candidate = f"{prefix}{side}_eef_pos"
    if candidate in replay_buffer.data:
        return candidate
    available = [k for k in replay_buffer.data.keys() if k.endswith("_eef_pos")]
    if not available:
        raise KeyError("No *_eef_pos keys found in replay buffer.")
    raise KeyError(
        f"Pose key '{candidate}' not found. Available *_eef_pos keys: {available}"
    )


def _resolve_rot_key(replay_buffer: ReplayBuffer, pos_key: str) -> Optional[str]:
    rot_key = pos_key.replace("_eef_pos", "_eef_rot_axis_angle")
    if rot_key in replay_buffer.data:
        return rot_key
    return None


def _load_dataset_plan(path: str) -> List[dict]:
    with open(os.path.expanduser(path), "rb") as f:
        plan = pickle.load(f)
    if not isinstance(plan, list):
        raise ValueError("Expected dataset_plan.pkl to contain a list of episodes.")
    return plan


def _extract_plan_pose(plan_episode: dict, side: str) -> Tuple[np.ndarray, np.ndarray]:
    key = f"grippers_{side}"
    if key not in plan_episode:
        raise KeyError(f"Dataset plan missing key '{key}'.")
    grippers = plan_episode[key]
    if not grippers:
        raise KeyError(f"No gripper data found for side '{side}'.")
    tcp_pose = grippers[0].get("tcp_pose")
    if tcp_pose is None:
        raise KeyError(f"Missing tcp_pose for side '{side}'.")
    pose = np.asarray(tcp_pose)
    if pose.ndim != 2 or pose.shape[1] < 6:
        raise ValueError("Expected tcp_pose with shape (T, 6).")
    pos = pose[:, :3]
    rot = pose[:, 3:6]
    return pos, rot


def _sample_axis_points(pose: np.ndarray, axis_length: float, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    origin = pose[:3, 3]
    axes = pose[:3, :3]
    ts = np.linspace(0.0, axis_length, samples, dtype=np.float32)
    axis_colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    points_list = []
    colors_list = []
    for axis_idx in range(3):
        direction = axes[:, axis_idx]
        pts = origin[None, :] + ts[:, None] * direction[None, :]
        points_list.append(pts)
        colors_list.append(np.tile(axis_colors[axis_idx], (samples, 1)))
    return np.concatenate(points_list, axis=0), np.concatenate(colors_list, axis=0)


def _colorize_by_z(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z_vals = points[:, 2]
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    denom = max(z_max - z_min, 1e-6)
    t = (z_vals - z_min) / denom
    colors = np.stack(
        [t, 1.0 - t, np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0.0, 1.0)], axis=1
    )
    return colors.astype(np.float32, copy=False)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    colors_uint8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors_uint8):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")


def _render_episode_start(
    replay_buffer: ReplayBuffer,
    episode_slice: slice,
    episode_name: str,
    pointmap_key: str,
    pos_keys: Dict[str, str],
    rot_keys: Dict[str, Optional[str]],
    output_dir: Path,
    axis_length: float,
    axis_samples: int,
) -> Optional[Path]:
    if pointmap_key not in replay_buffer.data:
        print(f"[alignment] Pointmap key '{pointmap_key}' missing; skipping render.")
        return None
    frame_idx = episode_slice.start
    pointmap = np.asarray(replay_buffer.data[pointmap_key][frame_idx])
    points = pointmap.reshape(-1, 3).astype(np.float32, copy=False)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    colors = _colorize_by_z(points)

    extra_points = []
    extra_colors = []
    for side, pos_key in pos_keys.items():
        rot_key = rot_keys.get(side)
        if rot_key is None:
            continue
        pos = np.asarray(replay_buffer.data[pos_key][frame_idx])
        rot = np.asarray(replay_buffer.data[rot_key][frame_idx])
        if not np.isfinite(pos).all() or not np.isfinite(rot).all():
            continue
        pose_vec = np.concatenate([pos, rot], axis=-1).astype(np.float32, copy=False)
        pose = pose_to_mat(pose_vec)
        axis_pts, axis_cols = _sample_axis_points(pose, axis_length, axis_samples)
        extra_points.append(axis_pts)
        extra_colors.append(axis_cols)

    if extra_points:
        extra_points_arr = np.concatenate(extra_points, axis=0)
        extra_colors_arr = np.concatenate(extra_colors, axis=0)
        points = np.concatenate([points, extra_points_arr], axis=0)
        colors = np.concatenate([colors, extra_colors_arr], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = episode_name.replace(os.sep, "_")
    ply_path = output_dir / f"{safe_name}_frame00000.ply"
    _write_ply(ply_path, points, colors)
    return ply_path


def _pose_mat_from_obs(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pose_vec = np.concatenate([pos, rot], axis=-1).astype(np.float32, copy=False)
    return pose_to_mat(pose_vec)


def _transform_pose_mat(
    pose_mat: np.ndarray, base_pose_mat: np.ndarray, pose_rep: str
) -> np.ndarray:
    return convert_pose_mat_rep(
        pose_mat, base_pose_mat=base_pose_mat, pose_rep=pose_rep, backward=False
    )


def _first_frame_abnormal(
    left_x: np.ndarray,
    right_x: np.ndarray,
    head_x: np.ndarray,
    threshold: float,
) -> Tuple[bool, float]:
    if len(left_x) == 0 or len(right_x) == 0 or len(head_x) == 0:
        return False, float("nan")
    lx, rx, hx = float(left_x[0]), float(right_x[0]), float(head_x[0])
    if not np.isfinite(lx + rx + hx):
        return False, float("nan")
    center_x = 0.5 * (lx + rx)
    error = abs(hx - center_x)
    return error > threshold, error


def _resolve_reference_side(
    comparison_frame: str, left_side: str, right_side: str, head_side: str
) -> Optional[str]:
    if comparison_frame == "world":
        return None
    if comparison_frame == "left":
        return left_side
    if comparison_frame == "right":
        return right_side
    if comparison_frame == "head":
        return head_side
    raise ValueError(f"Unsupported comparison frame '{comparison_frame}'")


def _positions_in_reference_frame(
    pose_mats: Dict[str, np.ndarray],
    reference_side: Optional[str],
    pose_rep: str,
) -> Dict[str, np.ndarray]:
    if reference_side is None:
        return {side: pose[:, :3, 3] for side, pose in pose_mats.items()}
    if reference_side not in pose_mats:
        raise KeyError(f"Reference side '{reference_side}' missing from pose data.")
    base_pose_mat = pose_mats[reference_side]
    return {
        side: _transform_pose_mat(pose, base_pose_mat, pose_rep)[:, :3, 3]
        for side, pose in pose_mats.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect demos where head x is not centered between grippers."
    )
    parser.add_argument(
        "replay_buffer_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to replay buffer (.zip or .zarr).",
    )
    parser.add_argument(
        "--dataset-plan",
        type=str,
        default=None,
        help="Path to dataset_plan.pkl; skips loading replay buffer.",
    )
    parser.add_argument("--prefix", type=str, default="gripper_", help="Prefix for eef pose keys.")
    parser.add_argument("--left-side", type=str, default="left", help="Left gripper side name.")
    parser.add_argument("--right-side", type=str, default="right", help="Right gripper side name.")
    parser.add_argument("--head-side", type=str, default="head", help="Head side name.")
    parser.add_argument("--left-key", type=str, default=None, help="Override left eef pos key.")
    parser.add_argument("--right-key", type=str, default=None, help="Override right eef pos key.")
    parser.add_argument("--head-key", type=str, default=None, help="Override head eef pos key.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Absolute x-axis deviation (meters) to flag as abnormal.",
    )
    parser.add_argument(
        "--comparison-frame",
        type=str,
        default="task",
        choices=["world", "head", "left", "right", "task"],
        help="Frame used for comparison (task uses task.reference_frame).",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="Task config for dataset rendering and task reference frame.",
    )
    parser.add_argument(
        "--render-dataset-pointmap",
        action="store_true",
        help="Render abnormal episodes via train_network/scripts/render_dataset_pointmap.py.",
    )
    parser.add_argument(
        "--render-dataset-path",
        type=str,
        default=None,
        help="Override dataset path for render_dataset_pointmap.py.",
    )
    parser.add_argument(
        "--render-reference-frame",
        type=str,
        default=None,
        help="Override reference frame for render_dataset_pointmap.py.",
    )
    parser.add_argument(
        "--render-output-dir",
        type=str,
        default=None,
        help="Output directory for render_dataset_pointmap.py PLY exports.",
    )
    parser.add_argument(
        "--render-pointmap-key",
        type=str,
        default="camera_head_pointmap",
        help="Pointmap key passed to render_dataset_pointmap.py.",
    )
    parser.add_argument(
        "--render-gripper-sides",
        type=str,
        nargs="*",
        default=None,
        help="Gripper prefixes to pass to render_dataset_pointmap.py (e.g. gripper_left).",
    )
    parser.add_argument(
        "--abnormal-list-out",
        type=str,
        default=None,
        help="Optional path to write abnormal episode names (one per line).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Max frame indices to print per episode.",
    )
    parser.add_argument(
        "--render-abnormal",
        action="store_true",
        help="Export a PLY pointmap for the first frame of abnormal episodes.",
    )
    parser.add_argument(
        "--pointmap-key",
        type=str,
        default="camera_head_pointmap",
        help="Pointmap key to use for rendering.",
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default=None,
        help="Output directory for rendered PLY files.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Length of gripper axes when rendering.",
    )
    parser.add_argument(
        "--axis-samples",
        type=int,
        default=20,
        help="Samples per axis line when rendering.",
    )
    args = parser.parse_args()

    use_dataset_plan = args.dataset_plan is not None
    replay_buffer = None
    pos_keys: Dict[str, str] = {}
    rot_keys: Dict[str, Optional[str]] = {}
    episode_names: List[str] = []
    plan = None

    if use_dataset_plan:
        if args.render_abnormal or args.render_dataset_pointmap:
            raise ValueError("Rendering requires a replay buffer; omit --dataset-plan.")
        plan = _load_dataset_plan(args.dataset_plan)
        episode_names = [
            str(episode.get("episode_name", idx)) for idx, episode in enumerate(plan)
        ]
    else:
        if not args.replay_buffer_path:
            parser.error("replay_buffer_path is required unless --dataset-plan is set.")
        store = _open_store(args.replay_buffer_path)
        replay_buffer = ReplayBuffer.create_from_group(zarr.group(store))

        left_key = _resolve_pose_key(
            replay_buffer, args.left_key, args.prefix, args.left_side
        )
        right_key = _resolve_pose_key(
            replay_buffer, args.right_key, args.prefix, args.right_side
        )
        head_key = _resolve_pose_key(
            replay_buffer, args.head_key, args.prefix, args.head_side
        )
        rot_keys = {
            args.left_side: _resolve_rot_key(replay_buffer, left_key),
            args.right_side: _resolve_rot_key(replay_buffer, right_key),
            args.head_side: _resolve_rot_key(replay_buffer, head_key),
        }
        pos_keys = {
            args.left_side: left_key,
            args.right_side: right_key,
            args.head_side: head_key,
        }
        episode_names = _decode_episode_names(replay_buffer.episode_names[:])

    task_cfg = None
    if args.task_config:
        task_cfg = _load_task_config(args.task_config)
    if args.comparison_frame == "task":
        if task_cfg is None:
            print(
                "[alignment] comparison-frame=task requested without --task-config; falling back to world."
            )
            args.comparison_frame = "world"
        else:
            args.comparison_frame = task_cfg.get("reference_frame", "world")

    pose_rep = "relative"
    if task_cfg is not None:
        pose_rep = task_cfg.get("pose_repr", {}).get("obs_pose_repr", pose_rep)

    reference_side = _resolve_reference_side(
        args.comparison_frame, args.left_side, args.right_side, args.head_side
    )

    render_script = None
    if args.render_dataset_pointmap:
        if not args.task_config:
            raise ValueError("--render-dataset-pointmap requires --task-config.")
        script_root = Path(__file__).resolve().parents[2]
        render_script = script_root / "train_network" / "scripts" / "render_dataset_pointmap.py"
        if not render_script.exists():
            raise FileNotFoundError(f"Render script missing at {render_script}")

    abnormal = []
    n_episodes = len(plan) if use_dataset_plan else replay_buffer.n_episodes
    for episode_idx in range(n_episodes):
        if use_dataset_plan:
            plan_episode = plan[episode_idx]
            left_pos, left_rot = _extract_plan_pose(plan_episode, args.left_side)
            right_pos, right_rot = _extract_plan_pose(plan_episode, args.right_side)
            head_pos, head_rot = _extract_plan_pose(plan_episode, args.head_side)
        else:
            episode_slice = replay_buffer.get_episode_slice(episode_idx)
            left_pos = np.asarray(replay_buffer.data[left_key][episode_slice])
            right_pos = np.asarray(replay_buffer.data[right_key][episode_slice])
            head_pos = np.asarray(replay_buffer.data[head_key][episode_slice])
            left_rot_key = rot_keys.get(args.left_side)
            right_rot_key = rot_keys.get(args.right_side)
            head_rot_key = rot_keys.get(args.head_side)
            if left_rot_key is None or right_rot_key is None or head_rot_key is None:
                raise KeyError("Missing rotation keys for grippers/head.")
            left_rot = np.asarray(replay_buffer.data[left_rot_key][episode_slice])
            right_rot = np.asarray(replay_buffer.data[right_rot_key][episode_slice])
            head_rot = np.asarray(replay_buffer.data[head_rot_key][episode_slice])

        if left_pos.ndim != 2 or right_pos.ndim != 2 or head_pos.ndim != 2:
            raise ValueError("Expected pose arrays with shape (T, 3).")
        if left_rot.ndim != 2 or right_rot.ndim != 2 or head_rot.ndim != 2:
            raise ValueError("Expected rotation arrays with shape (T, 3).")

        pose_mats = {
            args.left_side: _pose_mat_from_obs(left_pos, left_rot),
            args.right_side: _pose_mat_from_obs(right_pos, right_rot),
            args.head_side: _pose_mat_from_obs(head_pos, head_rot),
        }
        ref_positions = _positions_in_reference_frame(
            pose_mats, reference_side, pose_rep
        )
        left_ref = ref_positions[args.left_side]
        right_ref = ref_positions[args.right_side]
        head_ref = ref_positions[args.head_side]

        is_bad, error = _first_frame_abnormal(
            left_ref[:, 0], right_ref[:, 0], head_ref[:, 0], args.threshold
        )
        bad = np.array([0], dtype=int) if is_bad else np.array([], dtype=int)
        if bad.size > 0:
            name = episode_names[episode_idx] if episode_idx < len(episode_names) else str(episode_idx)
            abnormal.append((episode_idx, name, bad))
            print(
                f"[alignment] episode {episode_idx} ({name}) abnormal at frame 0 "
                f"(x error {error:.4f})"
            )
            if args.render_abnormal and replay_buffer is not None:
                render_dir = (
                    Path(args.render_dir)
                    if args.render_dir
                    else Path(args.replay_buffer_path).expanduser().resolve().parent
                    / "abnormal_alignment_pointmaps"
                )
                ply_path = _render_episode_start(
                    replay_buffer,
                    episode_slice,
                    name,
                    args.pointmap_key,
                    pos_keys,
                    rot_keys,
                    render_dir,
                    args.axis_length,
                    args.axis_samples,
                )
                if ply_path is not None:
                    print(f"[alignment] wrote {ply_path}")

            if args.render_dataset_pointmap and render_script is not None:
                dataset_path = args.render_dataset_path or args.replay_buffer_path
                render_ref = args.render_reference_frame or (
                    task_cfg.get("reference_frame") if task_cfg else None
                )
                render_out_dir = (
                    Path(args.render_output_dir)
                    if args.render_output_dir
                    else Path(args.replay_buffer_path).expanduser().resolve().parent
                    / "abnormal_dataset_pointmaps"
                )
                safe_name = name.replace(os.sep, "_")
                render_output = render_out_dir / f"{safe_name}.ply"
                first_bad = int(bad[0])
                cmd = [
                    sys.executable,
                    str(render_script),
                    args.task_config,
                    "--dataset-index",
                    str(episode_idx),
                    "--frame-index",
                    str(first_bad),
                    "--max-frames",
                    "1",
                    "--pointmap-key",
                    args.render_pointmap_key,
                    "--output",
                    str(render_output),
                ]
                if dataset_path:
                    cmd += ["--dataset-path", dataset_path]
                if render_ref:
                    cmd += ["--reference-frame", render_ref]
                if args.render_gripper_sides:
                    cmd += ["--gripper-sides", *args.render_gripper_sides]
                print(f"[alignment] rendering dataset pointmap: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

    if not abnormal:
        print("[alignment] No abnormal episodes found.")
    else:
        print(f"[alignment] Found {len(abnormal)} abnormal episodes.")

    if args.abnormal_list_out:
        out_path = Path(args.abnormal_list_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="ascii") as f:
            for _episode_idx, name, _bad in abnormal:
                f.write(f"{name}\n")
        print(f"[alignment] Wrote abnormal list to {out_path}")


if __name__ == "__main__":
    main()
