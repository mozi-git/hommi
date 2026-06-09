#!/usr/bin/env python3
"""Create a small synthetic HoMMI replay-buffer dataset for smoke tests.

The generated data is random and physically meaningless. Its only purpose is to
exercise the dataset loader, normalizer, model forward pass, and checkpoint path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

from hommi.common.replay_buffer import ReplayBuffer


def _smooth_pose(rng: np.random.Generator, n_frames: int, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = rng.normal(0.0, 0.002, size=(n_frames, 3)).astype(np.float32)
    pos = np.cumsum(steps, axis=0) + offset.astype(np.float32)
    rot = rng.normal(0.0, 0.03, size=(n_frames, 3)).astype(np.float32)
    return pos, rot


def _rgb(rng: np.random.Generator, n_frames: int, size: int) -> np.ndarray:
    base = rng.integers(0, 255, size=(n_frames, 1, 1, 3), dtype=np.uint8)
    noise = rng.integers(0, 32, size=(n_frames, size, size, 3), dtype=np.uint8)
    return (base + noise).astype(np.uint8)


def _pointmap(rng: np.random.Generator, n_frames: int, size: int) -> np.ndarray:
    grid = np.linspace(-0.3, 0.3, size, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    pointmap = np.empty((n_frames, size, size, 3), dtype=np.float16)
    for i in range(n_frames):
        z = np.full_like(xx, 1.0 + 0.02 * np.sin(i / 10.0))
        jitter = rng.normal(0.0, 0.002, size=(size, size, 3)).astype(np.float32)
        pointmap[i] = np.stack([xx, yy, z], axis=-1) + jitter
    return pointmap


def build_episode(
    rng: np.random.Generator,
    n_frames: int,
    rgb_size: int,
    pointmap_size: int,
) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}

    for side, offset in {
        "head": np.array([0.0, 0.0, 1.0]),
        "left": np.array([0.25, 0.18, 0.75]),
        "right": np.array([0.25, -0.18, 0.75]),
    }.items():
        pos, rot = _smooth_pose(rng, n_frames, offset)
        prefix = f"gripper_{side}"
        data[f"{prefix}_eef_pos"] = pos
        data[f"{prefix}_eef_rot_axis_angle"] = rot
        data[f"{prefix}_demo_start_pose"] = np.repeat(
            np.concatenate([pos[:1], rot[:1]], axis=-1), n_frames, axis=0
        ).astype(np.float32)
        data[f"{prefix}_demo_end_pose"] = np.repeat(
            np.concatenate([pos[-1:], rot[-1:]], axis=-1), n_frames, axis=0
        ).astype(np.float32)

    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[:, None]
    data["gripper_left_gripper_width"] = (0.04 + 0.02 * t).astype(np.float32)
    data["gripper_right_gripper_width"] = (0.05 - 0.015 * t).astype(np.float32)
    data["camera_head_lookatpoint"] = np.column_stack(
        [
            0.05 * np.sin(np.arange(n_frames, dtype=np.float32) / 12.0),
            np.zeros(n_frames, dtype=np.float32),
            np.ones(n_frames, dtype=np.float32),
        ]
    ).astype(np.float32)

    data["camera_head_main_rgb"] = _rgb(rng, n_frames, pointmap_size)
    data["camera_head_pointmap"] = _pointmap(rng, n_frames, pointmap_size)
    data["camera_left_main_rgb"] = _rgb(rng, n_frames, rgb_size)
    data["camera_right_main_rgb"] = _rgb(rng, n_frames, rgb_size)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/dummy_hommi_dataset.zarr.zip")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--rgb-size", type=int, default=224)
    parser.add_argument("--pointmap-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    rng = np.random.default_rng(args.seed)
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    for episode_idx in range(args.episodes):
        replay_buffer.add_episode(
            data=build_episode(rng, args.frames, args.rgb_size, args.pointmap_size),
            tasks=[
                {
                    "name": "dummy",
                    "start_idx": 0,
                    "end_idx": args.frames,
                    "labels": {},
                }
            ],
            compressors="disk",
            episode_name=f"dummy_episode_{episode_idx:03d}",
        )

    store = zarr.ZipStore(str(out_path), mode="w")
    try:
        replay_buffer.save_to_store(store, compressors="disk")
    finally:
        store.close()

    print(f"Wrote {out_path}")
    print(f"Episodes: {args.episodes}, frames per episode: {args.frames}")


if __name__ == "__main__":
    main()
