import argparse
import os
from typing import Sequence, Tuple

from hommi.demonstration_processing.utils.pointmap_alignment_viz import (
    PointmapVizOptions,
    render_pointmap_alignment_video,
)


def _parse_vec(arg: Sequence[float], expected_len: int) -> Tuple[float, ...]:
    values = tuple(float(x) for x in arg)
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} entries but received {len(values)}")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Render a pointmap/gripper alignment video for a demonstration.",
    )
    parser.add_argument("demonstration_dir", type=str, help="Path to the demo folder")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output video path. Defaults to <demo>/<depth_side>_pointmap_alignment.mp4",
    )
    parser.add_argument(
        "--depth-side",
        type=str,
        default="head",
        help="Camera side to visualize (default: head)",
    )
    parser.add_argument(
        "--gripper-sides",
        nargs="+",
        default=["left", "right"],
        help="Which gripper sides to render coordinate frames for.",
    )
    parser.add_argument(
        "--iphone-calibration",
        type=str,
        default="calibration/iphone15pro_calibration.yaml",
        help="Path to the calibration YAML containing the depth intrinsic matrix.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Visualize every Nth frame to speed up rendering.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap on the number of frames to render.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for the output video.")
    parser.add_argument(
        "--depth-shape",
        type=int,
        nargs=2,
        default=[192, 256],
        metavar=("H", "W"),
        help="Height/width used when decoding the raw depth file.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.02,
        help="Minimum depth to keep (meters). Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=1.2,
        help="Maximum depth to keep (meters). Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=48000,
        help="Maximum number of points to render per frame.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size passed to the Open3D renderer.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Length of the coordinate frame axes for the grippers.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("W", "H"),
        help="Output video resolution.",
    )
    parser.add_argument(
        "--background-color",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=("R", "G", "B", "A"),
        help="Renderer background color (RGBA).",
    )
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.45],
        help="Look-at target for the renderer camera.",
    )
    parser.add_argument(
        "--eye",
        type=float,
        nargs=3,
        default=[0.0, -0.28, -0.10],
        help="Camera eye position for the renderer.",
    )
    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        help="Camera up vector.",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.8,
        help="Multiplier applied to the automatically computed orbit radius.",
    )
    parser.add_argument(
        "--min-camera-radius",
        type=float,
        default=0.12,
        help="Minimum radius for the orbit camera to avoid degenerate views.",
    )
    parser.add_argument(
        "--lookat-marker-radius",
        type=float,
        default=0.02,
        help="Radius of the highlighted look-at point marker.",
    )
    parser.add_argument(
        "--use-rgb-colors",
        dest="use_rgb_colors",
        action="store_true",
        default=True,
        help="Colorize the point cloud using the RGB video instead of depth-based colors.",
    )
    parser.add_argument(
        "--disable-rgb-colors",
        dest="use_rgb_colors",
        action="store_false",
        help="Disable RGB coloring.",
    )
    parser.add_argument(
        "--merge-wrist-pointclouds",
        dest="merge_wrist_pointclouds",
        action="store_true",
        default=False,
        help="Merge wrist camera pointclouds into the head frame.",
    )
    parser.add_argument(
        "--no-merge-wrist-pointclouds",
        dest="merge_wrist_pointclouds",
        action="store_false",
        help="Disable merging wrist pointclouds.",
    )
    parser.add_argument(
        "--wrist-sides",
        nargs="+",
        default=["left", "right"],
        help="Wrist camera sides to merge when enabled.",
    )
    parser.add_argument(
        "--save-combined-pointmap",
        dest="save_combined_pointmap",
        action="store_true",
        default=False,
        help="Save combined pointmaps projected into the head image plane.",
    )
    parser.add_argument(
        "--no-save-combined-pointmap",
        dest="save_combined_pointmap",
        action="store_false",
        help="Disable combined pointmap saving.",
    )
    parser.add_argument(
        "--combined-pointmap-pattern",
        type=str,
        default="{depth_side}_combined_pointmap_{frame:05d}.ply",
        help="Output path template for combined pointmap PLYs (relative to demo dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render even if the output already exists.",
    )
    parser.add_argument(
        "--rotate-head-poses",
        action="store_true",
        help="Apply the same Z-rotation used in the dataset builder for head poses.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress information.",
    )

    args = parser.parse_args()
    demo_dir = os.path.abspath(args.demonstration_dir)
    output_path = args.output
    if output_path is None:
        filename = f"{args.depth_side}_pointmap_alignment.mp4"
        output_path = os.path.join(demo_dir, filename)

    min_depth = args.min_depth if args.min_depth > 0 else None
    max_depth = args.max_depth if args.max_depth > 0 else None

    options = PointmapVizOptions(
        iphone_calibration_path=args.iphone_calibration,
        depth_shape=tuple(args.depth_shape),
        depth_side=args.depth_side,
        gripper_sides=tuple(args.gripper_sides),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        fps=args.fps,
        min_depth=min_depth,
        max_depth=max_depth,
        max_points=args.max_points,
        point_size=args.point_size,
        axis_length=args.axis_length,
        renderer_width=args.render_size[0],
        renderer_height=args.render_size[1],
        background_color=_parse_vec(args.background_color, 4),
        lookat=_parse_vec(args.lookat, 3),
        eye=_parse_vec(args.eye, 3),
        up=_parse_vec(args.up, 3),
        camera_distance_scale=args.camera_distance_scale,
        min_camera_radius=args.min_camera_radius,
        rotate_head_poses=args.rotate_head_poses,
        lookat_marker_radius=args.lookat_marker_radius,
        use_rgb_colors=args.use_rgb_colors,
        merge_wrist_pointclouds=args.merge_wrist_pointclouds,
        wrist_sides=tuple(args.wrist_sides),
        save_combined_pointmap=args.save_combined_pointmap,
        combined_pointmap_pattern=args.combined_pointmap_pattern,
        skip_if_exists=not args.overwrite,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

    render_pointmap_alignment_video(
        demonstration_dir=demo_dir,
        output_path=os.path.abspath(output_path),
        options=options,
    )


if __name__ == "__main__":
    main()
