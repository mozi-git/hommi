"""Modified from UMI codebase for single-task episode sampling."""

from typing import Optional
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import scipy.spatial.transform as st
from hommi.common.replay_buffer import ReplayBuffer
from umi.common.pose_util import pose_to_mat

def get_val_mask(replay_buffer: ReplayBuffer, val_ratio, seed=0):
    """Build a validation mask over episodes."""
    n_segments = replay_buffer.n_episodes
    val_mask = np.zeros(n_segments, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_segments * val_ratio)), n_segments - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_segments, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        dataset_path: Optional[str]=None,
        mask: Optional[np.ndarray]=None,
        sparse_query_frequency_down_sample_steps: int=1,
        action_padding: bool=False,
        seed:int=0,
        discard_nan_pointmap_frames: bool=False,
        reference_frame: str="world",
        crop_pointcloud: bool=False,
        depth_range: tuple=(0.1, 0.8),
        gripper_length: float=0.13,
        pointmap_validity_num_workers: int=0,
    ):
        labels_keys = []

        # obs keys
        rgb_keys = list()
        lowdim_keys = list()
        pointmap_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim" and attr.get('in_replay_buffer', True):
                lowdim_keys.append(key)
            elif type == "pointmap":
                pointmap_keys.append(key)

        # horizon and down sample steps
        key_horizon = dict()
        key_down_sample_steps = dict()
        for key, attr in obs_shape_meta.items():
            key_horizon[key] = shape_meta['obs'][key]['horizon']
            key_down_sample_steps[key] = shape_meta['obs'][key]['down_sample_steps']

        key_horizon['action'] = shape_meta['action']['horizon']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        # prediction horizon
        num_pred_steps = shape_meta['action']['horizon']

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.pointmap_keys = pointmap_keys
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps
        self.mask = mask
        self.dataset_path = dataset_path
        if sparse_query_frequency_down_sample_steps < 1:
            raise ValueError("sparse_query_frequency_down_sample_steps must be >= 1")
        self.sparse_query_frequency_down_sample_steps = sparse_query_frequency_down_sample_steps
        self.action_padding = action_padding
        self.labels_keys = labels_keys
        self.num_pred_steps = num_pred_steps
        self.action_indexing_raw = dict()
        self.action_indexing = dict()
        self.discard_nan_pointmap_frames = discard_nan_pointmap_frames
        self.reference_frame = reference_frame
        self.crop_pointcloud = crop_pointcloud
        self.depth_range = depth_range
        self.gripper_length = gripper_length
        self.pointmap_validity_num_workers = pointmap_validity_num_workers
        self.valid_pointmap_frames = None
        self._pointmap_validity_keys = self._resolve_pointmap_validity_keys()


        self.load_replay_buffer_to_memory()
        if self.discard_nan_pointmap_frames and self.pointmap_keys:
            self.valid_pointmap_frames = self._compute_valid_pointmap_frames()
        self.setup_indices(seed)

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizer

    def load_replay_buffer_to_memory(self):
        # load low_dim to memory and keep rgb as compressed zarr array
        self.in_memory_replay_buffer = dict()
        self.num_robot = 0
        self.id_to_side = dict()
        for key in self.lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
                self.id_to_side[self.num_robot - 1] = key.split('_eef_pos')[0]  # e.g., 'robot0_eef_pos' -> 'robot0', 'gripper_left_eef_pos' -> 'gripper_left'

            if key.endswith('pos_abs'):
                axis = self.shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.in_memory_replay_buffer[key] = self.replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = self.shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = self.replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.in_memory_replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = self.shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = self.replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.in_memory_replay_buffer[key] = rot_out[:, list(axis)]
            else:
                if key in self.replay_buffer:
                    self.in_memory_replay_buffer[key] = self.replay_buffer[key][:]
        for key in self.rgb_keys:
            self.in_memory_replay_buffer[key] = self.replay_buffer[key]
        for key in self.pointmap_keys:
            self.in_memory_replay_buffer[key] = self.replay_buffer[key]

        if 'action' in self.replay_buffer:
            self.in_memory_replay_buffer['action'] = self.replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            # TODO: this is somewhat weird assumption to assume these key names and put this here (specific to UMI or general end effector pose control)
            actions = list()
            action_index_raw = 0
            action_index = 0
            self.in_memory_replay_buffer['action_raw'] = dict()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width', 'lookatpoint']:
                    if cat == 'lookatpoint' and self.id_to_side[robot_idx] == 'gripper_head':
                        key = 'camera_head_lookatpoint'
                    else:
                        key = f'{self.id_to_side[robot_idx]}_{cat}'
                    if key in self.in_memory_replay_buffer:
                        self.in_memory_replay_buffer['action_raw'][key] = self.replay_buffer[key][:]
                        if not self.shape_meta['obs'][key].get('ignore_by_action', False):
                            actions.append(self.in_memory_replay_buffer[key])
                            self.action_indexing_raw[key] = (action_index_raw, action_index_raw + self.in_memory_replay_buffer[key].shape[-1])
                            self.action_indexing[key] = (action_index, action_index + self.shape_meta['obs'][key]['shape'][-1])
                            action_index_raw += self.in_memory_replay_buffer[key].shape[-1]
                            action_index += self.shape_meta['obs'][key]['shape'][-1]
            self.in_memory_replay_buffer['action'] = np.concatenate(actions, axis=-1)

        # copy labels to memory
        for key in self.labels_keys:
            self.in_memory_replay_buffer[key] = self.replay_buffer.labels[key][:]

    def setup_indices(self, seed:int=0):
        # Prompting was removed for release cleanup; always use standard sampling.
        self.setup_indices_without_prompting()

    def setup_indices_without_prompting(self):
        segment_ends = self.replay_buffer.episode_ends[:]

        # create indices, including (current_within_segment_idx, episode_idx)
        start_of_segment_indices = {}
        end_of_segment_indices = {}
        indices = list()
        for i in range(len(segment_ends)):
            if self.mask is not None and not self.mask[i]:
                # skip episode
                continue
            end_data_idx = segment_ends[i]
            start_data_idx = 0 if i == 0 else segment_ends[i-1]
            episode_idx = i

            start_of_segment_indices[i] = len(indices)
            for current_data_idx in range(
                start_data_idx,
                end_data_idx,
                self.sparse_query_frequency_down_sample_steps,
            ):
                if self.valid_pointmap_frames is not None and not self._is_pointmap_window_valid(current_data_idx, start_data_idx):
                    continue
                if not self.action_padding and end_data_idx < current_data_idx + (self.key_horizon['action'] - 1) * self.key_down_sample_steps['action'] + 1:
                    continue

                current_within_segment_idx = current_data_idx - start_data_idx

                indices.append((current_within_segment_idx, episode_idx))
            end_of_segment_indices[i] = len(indices)

        self.indices = np.array(indices)

    def _compute_valid_pointmap_frames(self):
        cached = self._load_valid_pointmap_cache()
        if cached is not None:
            return cached
        num_steps = self.replay_buffer.n_steps
        valid = np.ones(num_steps, dtype=bool)
        chunk_size = 64
        chunks = [(start, min(start + chunk_size, num_steps)) for start in range(0, num_steps, chunk_size)]
        total = len(self._pointmap_validity_keys) * len(chunks)
        pbar = tqdm(total=total, desc="validating pointmaps", unit="chunk") if total > 0 else None

        def _finite_mask_for_chunk(args):
            key, start, end = args
            chunk = np.asarray(self.replay_buffer[key][start:end])
            finite = np.isfinite(chunk).reshape(end - start, -1).all(axis=1)
            return start, end, finite

        for key in self._pointmap_validity_keys:
            if key not in self.replay_buffer:
                print(f"[SequenceSampler] warning: validity key '{key}' missing; skipping.")
                continue
            if self.pointmap_validity_num_workers and self.pointmap_validity_num_workers > 0:
                work = [(key, start, end) for start, end in chunks]
                with ThreadPoolExecutor(max_workers=self.pointmap_validity_num_workers) as executor:
                    for start, end, finite in executor.map(_finite_mask_for_chunk, work):
                        valid[start:end] &= finite
                        if pbar is not None:
                            pbar.update(1)
            else:
                pointmap_arr = self.replay_buffer[key]
                for start, end in chunks:
                    chunk = np.asarray(pointmap_arr[start:end])
                    finite = np.isfinite(chunk).reshape(end - start, -1).all(axis=1)
                    valid[start:end] &= finite
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()
        dropped = int(np.count_nonzero(~valid))
        if dropped > 0:
            print(f"[SequenceSampler] dropping {dropped} frames with non-finite pointmap values.")
        if self.reference_frame != "world" and self.crop_pointcloud:
            valid = self._apply_pointmap_mask_validity(valid, chunk_size=chunk_size)
        self._save_valid_pointmap_cache(valid)
        return valid

    def _valid_pointmap_cache_path(self) -> Optional[Path]:
        if not self.dataset_path:
            return None
        dataset_path = Path(self.dataset_path).expanduser().resolve()
        return dataset_path.parent / f"{dataset_path.name}.pointmap_valid_frames.npz"

    def _pointmap_cache_meta(self) -> dict:
        return {
            "version": 1,
            "n_steps": int(self.replay_buffer.n_steps),
            "pointmap_keys": list(self.pointmap_keys),
            "reference_frame": self.reference_frame,
            "crop_pointcloud": self.crop_pointcloud,
            "depth_range": list(self.depth_range),
            "gripper_length": float(self.gripper_length),
        }

    def _load_valid_pointmap_cache(self) -> Optional[np.ndarray]:
        cache_path = self._valid_pointmap_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=False)
            meta_json = str(data["meta"].item())
            meta = json.loads(meta_json)
            if meta != self._pointmap_cache_meta():
                print(
                    f"[SequenceSampler] pointmap cache mismatch at {cache_path}; "
                    "recomputing validity."
                )
                return None
            valid = data["valid"]
            if valid.shape[0] != self.replay_buffer.n_steps:
                print(
                    f"[SequenceSampler] pointmap cache size mismatch at {cache_path}; "
                    "recomputing validity."
                )
                return None
            print(f"[SequenceSampler] loaded pointmap validity cache: {cache_path}")
            return valid.astype(bool, copy=False)
        except Exception as e:
            print(f"[SequenceSampler] failed to load pointmap cache {cache_path}: {e}")
            return None

    def _save_valid_pointmap_cache(self, valid: np.ndarray) -> None:
        cache_path = self._valid_pointmap_cache_path()
        if cache_path is None:
            return
        try:
            meta = self._pointmap_cache_meta()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                valid=valid.astype(bool, copy=False),
                meta=np.array(json.dumps(meta)),
            )
            print(f"[SequenceSampler] saved pointmap validity cache: {cache_path}")
        except Exception as e:
            print(f"[SequenceSampler] failed to save pointmap cache {cache_path}: {e}")
            return

    def _is_pointmap_window_valid(self, current_data_idx: int, start_data_idx: int) -> bool:
        for key in self.pointmap_keys:
            this_horizon = self.key_horizon[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            idx_to_sample = np.array(
                [current_data_idx - idx * this_downsample_steps for idx in range(this_horizon)]
            )
            idx_to_sample = idx_to_sample[::-1]
            idx_to_sample = idx_to_sample[idx_to_sample >= start_data_idx]
            if idx_to_sample.size == 0 or not np.all(self.valid_pointmap_frames[idx_to_sample]):
                return False
        return True

    def _resolve_pointmap_validity_keys(self) -> list[str]:
        keys = list(self.pointmap_keys)
        if self.reference_frame == "world":
            return keys
        if self.reference_frame not in {"left", "right", "head"}:
            print(f"[SequenceSampler] warning: unknown reference_frame '{self.reference_frame}', falling back to world.")
            return keys
        if self.crop_pointcloud:
            keys.extend([
                "gripper_head_eef_pos",
                "gripper_head_eef_rot_axis_angle",
                "gripper_left_eef_pos",
                "gripper_left_eef_rot_axis_angle",
                "gripper_right_eef_pos",
                "gripper_right_eef_rot_axis_angle",
            ])
        return keys

    def _apply_pointmap_mask_validity(self, valid: np.ndarray, chunk_size: int = 64) -> np.ndarray:
        pointmap_key = "camera_head_pointmap"
        if pointmap_key not in self.replay_buffer:
            print(f"[SequenceSampler] warning: pointmap key '{pointmap_key}' missing; skipping mask validity.")
            return valid

        required_keys = [
            "gripper_head_eef_pos",
            "gripper_head_eef_rot_axis_angle",
        ]
        for side in ["left", "right"]:
            required_keys.extend([
                f"gripper_{side}_eef_pos",
                f"gripper_{side}_eef_rot_axis_angle",
            ])
        missing = [key for key in required_keys if key not in self.in_memory_replay_buffer]
        if missing:
            print(f"[SequenceSampler] warning: missing pose keys for mask validity: {missing}")
            return valid

        num_steps = self.replay_buffer.n_steps
        chunks = [(start, min(start + chunk_size, num_steps)) for start in range(0, num_steps, chunk_size)]
        pbar = tqdm(chunks, desc="validating pointmap masks", unit="chunk")
        for start, end in pbar:
            if not np.any(valid[start:end]):
                continue
            raw_points = np.asarray(self.replay_buffer[pointmap_key][start:end])
            head_points = np.moveaxis(raw_points, -1, 1).astype(np.float32, copy=False)
            head_depth = head_points[:, 2:3, :, :]

            head_pose = np.concatenate([
                self.in_memory_replay_buffer["gripper_head_eef_pos"][start:end],
                self.in_memory_replay_buffer["gripper_head_eef_rot_axis_angle"][start:end],
            ], axis=-1)
            head_pose_mat = pose_to_mat(head_pose)

            ones = np.ones(head_points.shape[:1] + (1,) + head_points.shape[2:], dtype=head_points.dtype)
            points_homogeneous = np.concatenate([head_points, ones], axis=1)

            mask = (head_depth > self.depth_range[0]) & (head_depth < self.depth_range[1])
            for side in ["left", "right"]:
                side_pose = np.concatenate([
                    self.in_memory_replay_buffer[f"gripper_{side}_eef_pos"][start:end],
                    self.in_memory_replay_buffer[f"gripper_{side}_eef_rot_axis_angle"][start:end],
                ], axis=-1)
                side_pose_mat = pose_to_mat(side_pose)
                inv_side_pose_mat = np.linalg.inv(side_pose_mat)
                head_in_gripper_mat = np.einsum("bij,bjk->bik", inv_side_pose_mat, head_pose_mat)
                points_in_gripper = np.einsum("bij,bjhw->bihw", head_in_gripper_mat, points_homogeneous)
                mask &= (points_in_gripper[:, 2:3, :, :] > -self.gripper_length)

            has_any = mask.reshape(mask.shape[0], -1).any(axis=1)
            valid[start:end] &= has_any
        dropped = int(np.count_nonzero(~valid))
        if dropped > 0:
            dropped_idx = np.where(~valid)[0]
            sample = dropped_idx[:10]
            episode_idx = np.searchsorted(self.replay_buffer.episode_ends[:], sample, side="right")
            sample_pairs = list(zip(sample.tolist(), episode_idx.tolist()))
            print(
                "[SequenceSampler] dropping "
                f"{dropped} frames with empty pointmap masks after cropping; "
                f"sample (frame, episode)={sample_pairs}"
            )
        return valid

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx:int):
        """Sample one rollout chunk without prompting metadata."""
        return self.sample_sequence_without_prompting(idx)

    def sample_sequence_without_prompting(self, idx:int):
        """see `sample_sequence` for format"""
        current_within_segment_idx, episode_idx = self.indices[idx]
        result = {
            'metadata': {}
        }

        data_slice = self.replay_buffer.get_episode_slice(episode_idx)
        start_data_idx, end_data_idx = data_slice.start, data_slice.stop
        current_data_idx = start_data_idx + current_within_segment_idx

        result['metadata']['episode_idx'] = episode_idx

        obs_keys = self.rgb_keys + self.lowdim_keys + self.pointmap_keys
        if self.ignore_rgb_is_applied:
            # Skip only the RGB streams when collecting observations.
            obs_keys = self.lowdim_keys + self.pointmap_keys

        # observation
        for key in obs_keys:
            if key not in self.in_memory_replay_buffer:
                continue
            input_arr = self.in_memory_replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            is_key_upsampled = self.replay_buffer.is_key_upsampled(key)

            if key in self.rgb_keys:
                if is_key_upsampled:
                    # if key is upsampled, then we need to handle the horizon differently. In particular, we apply the horizon and downsampling at the frequency of the observed data.
                    # so if we have 10Hz ultrawide data (and 60Hz other data), then a horizon of 2 will find the two most recent ultrawide frames (note these frames will be different)
                    # this is different than just duplicating the 10Hz data to make it 60Hz and then getting the last 2 frames becuase in that case the two frames would likely be the same, while in our implementation we ensure those are two different frames since we operate on the sampling frequency of 10Hz (rather than 60Hz upsampled stream of the data)
                    # TODO: all the data is already aligned in post processing, so don't need to do this upsampling here, double check if this mess things up
                    upsample_current_data_idx = self.replay_buffer.map_upsample_index(key, current_data_idx)
                    upsample_start_data_idx = self.replay_buffer.map_upsample_index(key, start_data_idx)
                    num_valid = min(this_horizon, (upsample_current_data_idx - upsample_start_data_idx) // this_downsample_steps + 1)
                    upsample_slice_start = upsample_current_data_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[upsample_slice_start: upsample_current_data_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
                else:
                    num_valid = min(this_horizon, (current_data_idx - start_data_idx) // this_downsample_steps + 1)
                    slice_start = current_data_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[slice_start: current_data_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
            else:
                assert not is_key_upsampled, 'upsampling not yet supported for low dim data'
                idx_to_sample = np.array(
                    [current_data_idx - idx * this_downsample_steps for idx in range(this_horizon)])
                idx_to_sample = idx_to_sample[::-1] # flip to make smallest idx come first now
                idx_to_sample = [x for x in idx_to_sample if x >= start_data_idx]

                output = input_arr[idx_to_sample]

            # for all observation types pad with the first value
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)

            result[key] = output

        # action
        input_arr = self.in_memory_replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_data_idx, current_data_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_data_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output
        if 'action_raw' in self.in_memory_replay_buffer:
            input_arr = self.in_memory_replay_buffer['action_raw']
            output_raw = dict()
            for key in input_arr.keys():
                slice_end = min(end_data_idx, current_data_idx + (action_horizon - 1) * action_down_sample_steps + 1)
                output = input_arr[key][current_data_idx: slice_end: action_down_sample_steps]
                # solve padding
                if not self.action_padding:
                    assert output.shape[0] == action_horizon
                elif output.shape[0] < action_horizon:
                    padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
                    output = np.concatenate([output, padding], axis=0)
                output_raw[key] = output
            result['action_raw'] = output_raw

        return result

    def shuffle_data_ordering(self, seed:int):
        self.setup_indices(seed)

    def requires_epoch_shuffle(self) -> bool:
        return False

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply

