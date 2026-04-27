import os
from typing import Sequence

from hommi.demonstration_processing.utils.generic_util import (
    demonstration_to_display_string,
    get_demonstration_sides_present,
)
from hommi.demonstration_processing.utils.pointmap_alignment_viz import (
    PointmapVizOptions,
    render_pointmap_alignment_video,
)


def _as_tuple(values: Sequence, expected_len: int) -> tuple:
    assert len(values) == expected_len, (
        f"Expected {expected_len} values but received {len(values)} for {values}"
    )
    return tuple(values)


def visualize_pointmap_alignment(demonstration_iterator, cfg):
    """Hydra stage entry point that renders pointmap/gripper alignment videos."""
    options = PointmapVizOptions(
        iphone_calibration_path=cfg.iphone_calibration,
        depth_shape=tuple(cfg.depth_shape),
        depth_side=cfg.depth_side,
        gripper_sides=tuple(cfg.gripper_sides),
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        fps=cfg.fps,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth if cfg.max_depth > 0 else None,
        max_points=cfg.max_points,
        point_size=cfg.point_size,
        axis_length=cfg.axis_length,
        renderer_width=cfg.render_width,
        renderer_height=cfg.render_height,
        background_color=_as_tuple(cfg.background_color, 4),
        lookat=_as_tuple(cfg.lookat, 3),
        eye=_as_tuple(cfg.eye, 3),
        up=_as_tuple(cfg.up, 3),
        camera_distance_scale=cfg.camera_distance_scale,
        min_camera_radius=cfg.min_camera_radius,
        rotate_head_poses=cfg.rotate_head_poses,
        lookat_marker_radius=cfg.lookat_marker_radius,
        use_rgb_colors=cfg.use_rgb_colors,
        merge_wrist_pointclouds=cfg.merge_wrist_pointclouds,
        wrist_sides=tuple(cfg.wrist_sides),
        save_combined_pointmap=cfg.save_combined_pointmap,
        combined_pointmap_pattern=cfg.combined_pointmap_pattern,
        skip_if_exists=not cfg.overwrite,
        overwrite=cfg.overwrite,
        verbose=getattr(cfg, "verbose", False),
    )

    processed = 0
    skipped = 0
    failed = 0

    for demonstration_dir in demonstration_iterator("demonstration"):
        sides = get_demonstration_sides_present(demonstration_dir)
        if cfg.depth_side not in sides:
            continue

        display_name = demonstration_to_display_string(
            demonstration_dir, cfg.depth_side
        )
        output_name = cfg.output_video_name
        output_path = os.path.join(demonstration_dir, output_name)

        if os.path.exists(output_path) and not cfg.overwrite:
            skipped += 1
            continue

        print(f"{display_name} Rendering pointmap alignment to {output_path}")
        try:
            result_path = render_pointmap_alignment_video(
                demonstration_dir=demonstration_dir,
                output_path=output_path,
                options=options,
            )
        except Exception as exc:  # pragma: no cover - visualization errors
            print(f"{display_name} Failed to render pointmap alignment: {exc}")
            failed += 1
            continue

        if result_path is None:
            skipped += 1
        else:
            processed += 1

    print(
        f"[Pointmap Viz] processed={processed} skipped={skipped} failed={failed}"
    )
