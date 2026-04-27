import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import zarr

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    this_file = Path(__file__).resolve()
    added = False
    for parent in this_file.parents:
        candidate = parent / "deps" / "universal_manipulation_interface"
        if candidate.exists():
            sys.path.append(str(candidate))
            added = True
            break
    if not added:
        raise
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from hommi.common.replay_buffer import ReplayBuffer
from hommi.demonstration_processing.utils.pointmap_alignment_viz import (
    PointmapVizOptions,
    load_replay_buffer_sequence,
    render_pointmap_sequence,
    _load_intrinsic_matrix,
)

register_codecs()


def _parse_vec(arg, expected_len: int, cast=float) -> Tuple:
    values = tuple(cast(x) for x in arg)
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} entries but received {len(values)}")
    return values


def _open_store(path: str):
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return zarr.DirectoryStore(path)
    if path.endswith(".zip"):
        return zarr.ZipStore(path, mode="r")
    if path.endswith(".zarr"):
        return zarr.DirectoryStore(path)
    raise ValueError(f"Unsupported replay buffer path: {path}")


def _resolve_episode_index(
    replay_buffer: ReplayBuffer, episode_index: int, episode_name: str
) -> int:
    if episode_name:
        names = replay_buffer.episode_names[:]
        names_list = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names
        ]
        if episode_name not in names_list:
            raise ValueError(
                f"Episode name '{episode_name}' not found. Available: {names_list}"
            )
        return names_list.index(episode_name)
    if episode_index < 0 or episode_index >= replay_buffer.n_episodes:
        raise IndexError(
            f"Episode index {episode_index} out of bounds (n={replay_buffer.n_episodes})"
        )
    return episode_index


def main():
    parser = argparse.ArgumentParser(
        description="Render a replay-buffer pointmap visualization."
    )
    parser.add_argument("replay_buffer_path", type=str, help="Path to the replay buffer (.zarr or .zip).")
    parser.add_argument("--output", type=str, default=None, help="Optional output video path.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--episode-name", type=str, default=None, help="Episode name to visualize.")
    parser.add_argument("--depth-side", type=str, default="head", help="Camera side stored in the replay buffer.")
    parser.add_argument("--frame-stride", type=int, default=2, help="Render every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Cap the number of frames (-1 keeps all).")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for the output video.")
    parser.add_argument("--depth-shape", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--min-depth", type=float, default=0.02, help="Minimum depth to keep (meters).")
    parser.add_argument("--max-depth", type=float, default=1.2, help="Maximum depth to keep (meters).")
    parser.add_argument("--max-points", type=int, default=48000, help="Maximum number of points per frame.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Rendered point size.")
    parser.add_argument("--axis-length", type=float, default=0.08, help="Length of pose axes.")
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("W", "H"),
        help="Renderer resolution.",
    )
    parser.add_argument(
        "--background-color",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=("R", "G", "B", "A"),
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.8,
        help="Scale factor for automatic orbit camera distance.",
    )
    parser.add_argument(
        "--min-camera-radius",
        type=float,
        default=0.12,
        help="Minimum orbit radius to avoid degeneracy.",
    )
    parser.add_argument(
        "--lookat-marker-radius",
        type=float,
        default=0.02,
        help="Radius of the look-at marker sphere.",
    )
    parser.add_argument(
        "--iphone-calibration",
        type=str,
        default="calibration/iphone15pro_calibration.yaml",
        help="Calibration YAML containing depth intrinsics.",
    )
    parser.add_argument(
        "--reference-frame",
        type=str,
        default="head",
        choices=["head", "gripper"],
        help="Frame used to express points (head or specific gripper).",
    )
    parser.add_argument(
        "--reference-side",
        type=str,
        default="right",
        help="Gripper side to use when --reference-frame=gripper.",
    )
    parser.add_argument(
        "--use-rgb-colors",
        dest="use_rgb_colors",
        action="store_true",
        default=True,
        help="Colorize using synchronized RGB frames.",
    )
    parser.add_argument(
        "--disable-rgb-colors",
        dest="use_rgb_colors",
        action="store_false",
        help="Disable RGB coloring.",
    )
    parser.add_argument(
        "--save-combined-pointmap",
        dest="save_combined_pointmap",
        action="store_true",
        default=False,
        help="Save a combined PLY pointmap for the first rendered frame.",
    )
    parser.add_argument(
        "--no-save-combined-pointmap",
        dest="save_combined_pointmap",
        action="store_false",
        help="Disable combined pointmap export.",
    )
    parser.add_argument(
        "--combined-pointmap-pattern",
        type=str,
        default="{depth_side}_combined_pointmap_{frame:05d}.ply",
        help="Filename template for combined pointmaps.",
    )
    parser.add_argument(
        "--save-pointmap-dir",
        type=str,
        default=None,
        help="Directory for saving combined pointmaps (defaults to replay buffer directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output video if it already exists.",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="avc1",
        help="FourCC video codec (default: avc1 for H.264).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    options = PointmapVizOptions(
        iphone_calibration_path=args.iphone_calibration,
        depth_shape=tuple(args.depth_shape),
        depth_side=args.depth_side,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        fps=args.fps,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_points=args.max_points,
        point_size=args.point_size,
        axis_length=args.axis_length,
        renderer_width=args.render_size[0],
        renderer_height=args.render_size[1],
        background_color=_parse_vec(args.background_color, 4),
        camera_distance_scale=args.camera_distance_scale,
        min_camera_radius=args.min_camera_radius,
        lookat_marker_radius=args.lookat_marker_radius,
        use_rgb_colors=args.use_rgb_colors,
        save_combined_pointmap=args.save_combined_pointmap,
        combined_pointmap_pattern=args.combined_pointmap_pattern,
        overwrite=args.overwrite,
        verbose=args.verbose,
        reference_frame=args.reference_frame,
        reference_side=args.reference_side,
        video_fourcc=args.fourcc,
    )

    rb_path = Path(args.replay_buffer_path)
    out_path = args.output
    if out_path is None:
        stem = rb_path.stem.replace(".zarr", "")
        out_dir = rb_path.parent
        out_path = out_dir / f"{stem}_ep{args.episode_index:04d}_pointmap.mp4"
    out_path = os.path.abspath(str(out_path))

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")

    store = _open_store(args.replay_buffer_path)
    try:
        root = zarr.group(store)
        replay_buffer = ReplayBuffer.create_from_group(root)
        episode_idx = _resolve_episode_index(
            replay_buffer, args.episode_index, args.episode_name
        )
        sequence = load_replay_buffer_sequence(
            replay_buffer, episode_idx, options, camera_side=args.depth_side
        )
        intrinsic = _load_intrinsic_matrix(options.iphone_calibration_path)
        total_frames = len(sequence.head_poses)
        if total_frames == 0:
            raise RuntimeError("Selected episode has no frames.")
        max_frames = total_frames if options.max_frames <= 0 else min(total_frames, options.max_frames)
        frame_indices = list(range(0, max_frames, options.frame_stride))
        if not frame_indices:
            raise RuntimeError("No frames selected after applying stride/max_frames.")

        episode_name = replay_buffer.get_episode_name(episode_idx)
        if isinstance(episode_name, bytes):
            episode_name = episode_name.decode("utf-8")
        display_name = f"{rb_path.name}::{episode_name}"

        save_root = args.save_pointmap_dir
        if save_root is None:
            save_root = os.path.dirname(os.path.abspath(args.replay_buffer_path))

        render_pointmap_sequence(
            display_name=display_name,
            sequence=sequence,
            intrinsic=intrinsic,
            output_path=out_path,
            frame_indices=frame_indices,
            options=options,
            save_root=save_root,
        )
        print(f"[replay_pointmap] Wrote visualization to {out_path}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
