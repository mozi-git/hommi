import copy
from typing import Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
from typing import Callable, List

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from hommi.common.replay_buffer import ReplayBuffer
from hommi.common.generic_util import get_identity_normalizer_from_shape
from hommi.train_network.common.sampler import get_val_mask
from hommi.train_network.common.sampler import SequenceSampler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d
from hommi.train_network.common.augmentation import ImageAugmentation
from hommi.train_network.dataset.base_dataset import BaseDataset

register_codecs()

class StreamingArrayStats:
    def __init__(self, replay_buffer: Optional[ReplayBuffer]=None):
        self._min = None
        self._max = None
        self.H = None
        self.W = None
        self._nan_debug_count = 0
        self._nan_debug_limit = 5
        self.replay_buffer = replay_buffer

    def update(self, batch, mask=None, episode_idx=None, episode_name=None, key_name: Optional[str]=None):
        if torch.is_tensor(batch):
            batch = batch.detach().cpu().numpy()
        # batch: (B, T, C, H, W), get min and max over B, T, H, W of each channel
        if mask is not None:
            # mask: (B, T, 1, H, W)
            batch = batch.copy()
            mask = mask.repeat(1, 1, batch.shape[2], 1, 1)
            batch[mask == 0] = np.nan
        if episode_idx is not None and self._nan_debug_count < self._nan_debug_limit:
            if torch.is_tensor(episode_idx):
                episode_idx = episode_idx.detach().cpu().numpy()
            all_nan = np.all(np.isnan(batch), axis=(2, 3, 4))
            if np.any(all_nan):
                bad = np.argwhere(all_nan)
                sample = bad[:10]
                if episode_name is not None:
                    sample_pairs = [
                        (int(episode_idx[b]), str(episode_name[b]), int(t))
                        for b, t in sample
                    ]
                else:
                    sample_pairs = [(int(episode_idx[b]), int(t)) for b, t in sample]
                name = key_name or "pointmap"
                print(
                    f"[UmiTaskDataset] normalizer NaN frame(s) detected for {name}; "
                    f"sample (episode_idx, episode_name, t)={sample_pairs}"
                )
                self._nan_debug_count += 1
        B, T, C, H, W = batch.shape
        if self.H is None:
            self.H = H
            self.W = W
        # batch = batch.moveaxis(2, -1).reshape(B * T * H * W, C)  # (B*T*H*W, C)
        batch = batch.reshape(B * T, C, H * W).transpose(0, 2, 1).reshape(B * T * H * W, C)  # (B*T*H*W, C)
        if self._min is None:
            dims = batch.shape[-1]
            self._min = np.full(dims, np.inf, dtype=np.float32)
            self._max = np.full(dims, -np.inf, dtype=np.float32)

        batch_min = np.nanmin(batch, axis=0).astype(np.float32, copy=False)
        batch_max = np.nanmax(batch, axis=0).astype(np.float32, copy=False)
        self._min = np.minimum(self._min, batch_min)
        self._max = np.maximum(self._max, batch_max)

    def to_stats(self):
        return {
            # expand from (C,) to (C, H, W)
            'min': self._min.repeat(self.H * self.W).reshape(-1, self.H, self.W),
            'max': self._max.repeat(self.H * self.W).reshape(-1, self.H, self.W),
        }

class UmiTaskDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        sparse_query_frequency_down_sample_steps: int=1,
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        normalizer_batch_size: int=4,
        normalizer_num_workers: int=0,
        normalizer_pin_memory: bool=False,
        normalizer_persistent_workers: bool=False,
        seed: int=42,
        val_ratio: float=0.0,
        reference_frame: str='world',    # 'world' or 'head' or 'left' or 'right'
        crop_pointcloud: bool=False,    # whether to crop pointcloud (crop out points behind the grippers)
        depth_range: tuple=(0.1, 0.8),   # min and max depth for pointcloud cropping
        gripper_length: float=0.13,
        discard_nan_pointmap_frames: bool=False,
        pointmap_validity_num_workers: int=0,
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        self.reference_frame = reference_frame
        self.crop_pointcloud = crop_pointcloud
        self.depth_range = depth_range
        self.gripper_length = gripper_length
        self.discard_nan_pointmap_frames = discard_nan_pointmap_frames
        self.pointmap_validity_num_workers = pointmap_validity_num_workers
        self.dataset_path = dataset_path
        self.sparse_query_frequency_down_sample_steps = sparse_query_frequency_down_sample_steps
        self.normalizer_batch_size = normalizer_batch_size
        self.normalizer_num_workers = normalizer_num_workers
        self.normalizer_pin_memory = normalizer_pin_memory
        self.normalizer_persistent_workers = normalizer_persistent_workers
        
        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )

        # print replay buffer info
        print(f'[UmiTaskDataset] replay buffer has {replay_buffer.n_episodes} episodes')
        for key in replay_buffer.keys():
            print(f'[UmiTaskDataset] replay buffer key {key} has shape {replay_buffer[key].shape}')

        self.num_robot = 0
        self.id_to_side = dict()
        rgb_keys = list()
        lowdim_keys = list()
        pointmap_keys = list()
        for key, attr in shape_meta['obs'].items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'pointmap':
                pointmap_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1
                self.id_to_side[self.num_robot - 1] = key.split('_eef_pos')[0]  # e.g., 'robot0_eef_pos' -> 'robot0', 'gripper_left_eef_pos' -> 'gripper_left'

        val_mask = get_val_mask(
            replay_buffer=replay_buffer,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.pointmap_keys = pointmap_keys
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.is_training_dataset = True
        self.sampler_kwargs = {
            'shape_meta': self.shape_meta,
            'replay_buffer': self.replay_buffer,
            'dataset_path': self.dataset_path,
            'action_padding': self.action_padding,
            'sparse_query_frequency_down_sample_steps': self.sparse_query_frequency_down_sample_steps,
            'seed': seed,
            'discard_nan_pointmap_frames': self.discard_nan_pointmap_frames,
            'reference_frame': self.reference_frame,
            'crop_pointcloud': self.crop_pointcloud,
            'depth_range': self.depth_range,
            'gripper_length': self.gripper_length,
            'pointmap_validity_num_workers': self.pointmap_validity_num_workers,
        }

        sampler = SequenceSampler(
            mask=train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler
        self.action_indexing_raw = sampler.action_indexing_raw
        self.action_indexing = sampler.action_indexing
        print(f'[UmiTaskDataset] action indexing raw: {self.action_indexing_raw}')
        print(f'[UmiTaskDataset] action indexing: {self.action_indexing}')

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            mask=self.val_mask,
            **self.sampler_kwargs
        )

        val_set.is_training_dataset = False
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        pointmap_accumulators = {
            key: StreamingArrayStats(replay_buffer=self.replay_buffer)
            for key in self.pointmap_keys
        }
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=self.normalizer_batch_size,
            num_workers=self.normalizer_num_workers,
            pin_memory=self.normalizer_pin_memory,
            persistent_workers=self.normalizer_persistent_workers and self.normalizer_num_workers > 0,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            episode_idx = batch['metadata'].get('episode_idx', None)
            episode_name = None
            if episode_idx is not None:
                if torch.is_tensor(episode_idx):
                    episode_idx_np = episode_idx.detach().cpu().numpy()
                else:
                    episode_idx_np = np.asarray(episode_idx)
                episode_name = [self.replay_buffer.get_episode_name(int(idx)) for idx in episode_idx_np]
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            for key in self.pointmap_keys:
                pointmap_accumulators[key].update(
                    batch['obs'][key],
                    mask=batch['obs'].get(f'{key}_mask', None),
                    episode_idx=episode_idx,
                    episode_name=episode_name,
                    key_name=key,
                )
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            print(f"data cache {key} {data_cache[key].shape}")
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        # assert data_cache['action'].shape[-1] % self.num_robot == 0
        # dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        # for i in range(self.num_robot):
        #     action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
        #     action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
        #     action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper
        # using action indexing
        for key, (start, end) in self.action_indexing.items():
            if key.endswith('eef_pos'):
                action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., start:end])))
            elif key.endswith('eef_rot_axis_angle'):
                action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., start:end])))
            elif key.endswith('gripper_width'):
                action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., start:end])))
            elif key.endswith('lookatpoint'):
                action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., start:end])))
            else:
                raise RuntimeError(f'unsupported action key {key}')

        normalizer['action'] = concatenate_normalizer(action_normalizers)
        print(f'[UmiTaskDataset] action normalizer stats: {normalizer["action"].get_input_stats().max, normalizer["action"].get_input_stats().min}')

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if self.shape_meta['obs'][key]['ignore_by_policy']:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('lookatpoint'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
            print(f'[UmiTaskDataset] obs {key} normalizer: {normalizer[key].get_input_stats().max, normalizer[key].get_input_stats().min}')

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
            print(f'[UmiTaskDataset] image {key} normalizer: {normalizer[key].get_input_stats().max, normalizer[key].get_input_stats().min}')

        # pointmap
        for key in self.pointmap_keys:
            # normalizer[key] = get_identity_normalizer_from_shape(self.shape_meta['obs'][key]['shape'])
            stat = pointmap_accumulators[key].to_stats()
            normalizer[key] = get_range_normalizer_from_stat(stat)
            normalizer[f'{key}_mask'] = get_identity_normalizer_from_shape((1, self.shape_meta['obs'][key]['shape'][1], self.shape_meta['obs'][key]['shape'][2]))
            print(f'[UmiTaskDataset] pointmap {key} normalizer: {normalizer[key].get_input_stats().max, normalizer[key].get_input_stats().min}')
        del dataloader
        del data_cache
        return normalizer
    
    def shuffle_data_ordering(self, seed:int):
        self.sampler.shuffle_data_ordering(seed)

    def requires_epoch_shuffle(self) -> bool:
        return self.sampler.requires_epoch_shuffle()

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # print(f'[UmiTaskDataset] sampling idx {idx} from sampler')
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # clamp to [0,1]
            obs_dict[key] = np.clip(obs_dict[key], 0.0, 1.0)
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            if self.shape_meta['obs'][key].get('in_replay_buffer', True) and key in data:
                obs_dict[key] = data[key].astype(np.float32)
                del data[key]

        for key in self.pointmap_keys:
            if key in data:
                obs_dict[key] = np.moveaxis(data[key], -1, 1)   # T, C, H, W
                del data[key]
        
        # generate relative pose between ees
        for key in self.lowdim_keys:
            if key.endswith('eef_pos'):
                pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[key],
                    obs_dict[key.replace('eef_pos', 'eef_rot_axis_angle')]
                ], axis=-1))
                # find keys that start with f'{key}_wrt'
                for other_key in self.lowdim_keys:
                    if other_key.startswith(f'{key}_wrt'):
                        other_side = other_key.split('_')[-1]
                        other_pose_mat = pose_to_mat(np.concatenate([
                            obs_dict[key.replace(key.split('_')[1], other_side)],
                            obs_dict[key.replace('eef_pos', 'eef_rot_axis_angle').replace(key.split('_')[1], other_side)]
                        ], axis=-1))
                        rel_obs_pose_mat = np.stack([
                            np.linalg.inv(other_pose_mat[i, :]) @ pose_mat[i, :] for i in range(pose_mat.shape[0])
                        ], axis=0)
                        # rel_obs_pose_mat = convert_pose_mat_rep(
                        #     pose_mat,
                        #     base_pose_mat=other_pose_mat[-1],
                        #     pose_rep='relative',
                        #     backward=False)
                        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                        obs_dict[other_key] = rel_obs_pose[:,:3]
                        obs_dict[other_key.replace('eef_pos', 'eef_rot_axis_angle')] = rel_obs_pose[:,3:]
        # for robot_id in range(self.num_robot):
        #     # convert pose to mat
        #     pose_mat = pose_to_mat(np.concatenate([
        #         obs_dict[f'robot{robot_id}_eef_pos'],
        #         obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
        #     ], axis=-1))
        #     for other_robot_id in range(self.num_robot):
        #         if robot_id == other_robot_id:
        #             continue
        #         if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
        #             continue
        #         other_pose_mat = pose_to_mat(np.concatenate([
        #             obs_dict[f'robot{other_robot_id}_eef_pos'],
        #             obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
        #         ], axis=-1))
        #         rel_obs_pose_mat = convert_pose_mat_rep(
        #             pose_mat,
        #             base_pose_mat=other_pose_mat[-1],
        #             pose_rep='relative',
        #             backward=False)
        #         rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        #         obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
        #         obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
                
        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        # transform poses into the requested reference frame if not 'in world' frame
        # for example, if using 'left' frame, everything is expressed in the t=0 left gripper frame
        if self.reference_frame == 'world':
            # compute action and eef pose in requested (likely relative) representation (starting from absolute)
            actions = list()
            for robot_id in range(self.num_robot):
                if f'{self.id_to_side[robot_id]}_eef_pos' in self.action_indexing_raw and f'{self.id_to_side[robot_id]}_eef_rot_axis_angle' in self.action_indexing_raw:
                    # convert pose to mat
                    pose_mat = pose_to_mat(np.concatenate([
                        obs_dict[f'{self.id_to_side[robot_id]}_eef_pos'],
                        obs_dict[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle']
                    ], axis=-1))
                    # action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])  
                    if self.shape_meta['obs'][f'{self.id_to_side[robot_id]}_eef_pos'].get('ignore_by_action', False) or self.shape_meta['obs'][f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'].get('ignore_by_action', False):
                        action_pose = None
                    else:
                        action_mat = pose_to_mat(np.concatenate([data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_pos'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_pos'][1]],
                                                                data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][1]],
                                                                ], axis=-1))
                        action_pose_mat = convert_pose_mat_rep(
                            action_mat, 
                            base_pose_mat=pose_mat[-1],
                            pose_rep=self.obs_pose_repr,
                            backward=False)
                        action_pose = mat_to_pose10d(action_pose_mat)

                    # solve relative obs
                    obs_pose_mat = convert_pose_mat_rep(
                        pose_mat, 
                        base_pose_mat=pose_mat[-1],
                        pose_rep=self.obs_pose_repr,
                        backward=False)
                
                    # convert pose to pos + rot6d representation
                    obs_pose = mat_to_pose10d(obs_pose_mat)
                else:
                    obs_pose = None
                    action_pose = None

                # solve look at point (should be in the base egocentric frame)
                if self.id_to_side[robot_id] == 'gripper_head' and 'camera_head_lookatpoint' in self.lowdim_keys and 'camera_head_lookatpoint' in self.action_indexing_raw:
                    # Get the look at point from data (in head's egocentric frame)
                    lookatpoint_action = data['action'][..., self.action_indexing_raw['camera_head_lookatpoint'][0]:self.action_indexing_raw['camera_head_lookatpoint'][1]]  # shape: (action_horizon, 3)
                    lookatpoint_obs = obs_dict['camera_head_lookatpoint']   # (obs_horizon, 3)

                    pose_mat = pose_to_mat(np.concatenate([
                        obs_dict[f'{self.id_to_side[robot_id]}_eef_pos'],
                        obs_dict[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle']
                    ], axis=-1))    # (obs_horizon, 4, 4)
                    # action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])  
                    action_mat = pose_to_mat(np.concatenate([data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_pos'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_pos'][1]],
                                                            data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][1]],
                                                            ], axis=-1))    # (action_horizon, 4, 4)
                    
                    # print(f"lookatpoint_action {lookatpoint_action.shape}, lookatpoint_obs {lookatpoint_obs.shape}, pose_mat {pose_mat.shape}, action_mat {action_mat.shape}")
                    
                    # Convert all points to homogeneous coordinates at once
                    lookatpoint_action_homogeneous = np.concatenate([
                        lookatpoint_action, 
                        np.ones((lookatpoint_action.shape[0], 1))
                    ], axis=-1)  # shape: (action_horizon, 4)
                    lookatpoint_obs_homogeneous = np.concatenate([
                        lookatpoint_obs,
                        np.ones((lookatpoint_obs.shape[0], 1))
                    ], axis=-1)  # shape: (obs_horizon, 4)
                    base_pose_inv = np.linalg.inv(pose_mat[-1])  # (4, 4)
                    
                    # Batch transform action lookatpoints
                    # Step 1: Transform from head egocentric to world frame using einsum
                    points_world_action = np.einsum('tij,tj->ti', action_mat, lookatpoint_action_homogeneous)  # (action_horizon, 4)
                    # Step 2: Transform from world to base frame using broadcasting
                    points_base_action = (base_pose_inv @ points_world_action.T).T  # (action_horizon, 4)
                    action_lookatpoint = points_base_action[:, :3]
                    
                    # Batch transform observation lookatpoints
                    # Step 1: Transform from head egocentric to world frame using einsum
                    points_world_obs = np.einsum('tij,tj->ti', pose_mat, lookatpoint_obs_homogeneous)  # (obs_horizon, 4)
                    # Step 2: Transform from world to base frame using broadcasting
                    points_base_obs = (base_pose_inv @ points_world_obs.T).T  # (obs_horizon, 4)
                    obs_dict['camera_head_lookatpoint'] = points_base_obs[:, :3]
                else:
                    action_lookatpoint = None
            
                # action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
                if self.id_to_side[robot_id]+'_gripper_width' in self.action_indexing_raw:
                    action_gripper = data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_gripper_width'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_gripper_width'][1]]
                else:
                    action_gripper = None

                if action_pose is not None:
                    if action_gripper is not None:
                        actions.append(np.concatenate([action_pose, action_gripper], axis=-1))
                    else:
                        actions.append(action_pose)
                else:
                    # if no gripper width is provided, we assume the action is just the pose (for the head side)
                    if action_lookatpoint is not None:
                        actions.append(action_lookatpoint)

                if obs_pose is not None:
                    # generate data
                    obs_dict[f'{self.id_to_side[robot_id]}_eef_pos'] = obs_pose[:,:3]
                    obs_dict[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'] = obs_pose[:,3:]

            data['action'] = np.concatenate(actions, axis=-1)       # TODO: assemble the actions according to action_indexing
        else:
            # initiate action array (according to self.action_indexing)
            actions = np.zeros((data['action'].shape[0], sum([end - start for start, end in self.action_indexing.values()])), dtype=np.float32)
            assert self.reference_frame == 'left' or self.reference_frame == 'right' or self.reference_frame == 'head', f'unsupported reference frame {self.reference_frame}'
            base_pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'gripper_{self.reference_frame}_eef_pos'],
                obs_dict[f'gripper_{self.reference_frame}_eef_rot_axis_angle']
            ], axis=-1))
            # transform pointmap and lookatpoint as well (assuming we only use head pointmaps/lookatpoints, which are in the head egocentric frame not the world frame)
            if len(self.pointmap_keys) > 0 or ('camera_head_lookatpoint' in self.lowdim_keys):
                # get head pose transform relative to the base frame
                head_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict['gripper_head_eef_pos'],
                    obs_dict['gripper_head_eef_rot_axis_angle']
                ], axis=-1))
                ref_head_pose_mat = convert_pose_mat_rep(
                    head_pose_mat,
                    base_pose_mat=base_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)  # (T, 4, 4)
            for key in self.pointmap_keys:
                assert key == 'camera_head_pointmap', f'only camera_head_pointmap supported for pointmap transform, got {key}'
                head_points = obs_dict[key]
                # convert all points to homogeneous coordinates at once
                points_homogeneous = np.concatenate([
                    head_points,      # (T, 3, H, W)
                    np.ones(head_points.shape[:1]+(1,)+head_points.shape[2:])   # (T, 1, H, W)
                ], axis=1)  # shape: (T, 4, H, W)
                # multiply with ref head pose mat
                points_transformed = np.einsum('tij,tjkm->tikm', ref_head_pose_mat, points_homogeneous)  # (T, 4, H, W)
                mask = np.ones(points_homogeneous.shape[:1]+(1,)+points_homogeneous.shape[2:], dtype=bool)   # (T, 1, H, W)
                if self.crop_pointcloud:
                    # crop out points outside depth range in the original head camera frame (depth resides there)
                    head_depth = head_points[:, 2:3, :, :]
                    mask &= (head_depth > self.depth_range[0]) & (head_depth < self.depth_range[1])
                    # further crop out points that are inside the other gripper's tcp frame
                    for side in ['left', 'right']:
                        side_pose_mat = pose_to_mat(np.concatenate([
                            obs_dict[f'gripper_{side}_eef_pos'],
                            obs_dict[f'gripper_{side}_eef_rot_axis_angle']
                        ], axis=-1))
                        head_in_gripper_mat = np.stack([
                            np.linalg.inv(side_pose_mat[t]) @ head_pose_mat[t] for t in range(head_pose_mat.shape[0])
                        ], axis=0)   # (T, 4, 4)
                        points_in_gripper = np.einsum('tij,tjkm->tikm', head_in_gripper_mat, points_homogeneous)  # (T, 4, H, W)
                        # crop out points with z < -0.13 in the gripper tcp frame
                        mask &= (points_in_gripper[:, 2:3, :, :] > -self.gripper_length)
                obs_dict[f'{key}_mask'] = mask.astype(bool)
                # points_transformed = points_transformed * mask
                obs_dict[key] = points_transformed[:, :3, :, :]   # (T, 3, H, W)
            if 'camera_head_lookatpoint' in self.lowdim_keys:
                lookatpoint_obs = obs_dict['camera_head_lookatpoint']   # (obs_horizon, 3)
                lookartpoint_obs_homogeneous = np.concatenate([
                    lookatpoint_obs,
                    np.ones((lookatpoint_obs.shape[0], 1))
                ], axis=-1)  # shape: (obs_horizon, 4)
                lookatpoint_obs_transformed = np.einsum('tij,tj->ti', ref_head_pose_mat, lookartpoint_obs_homogeneous)  # (obs_horizon, 4)
                obs_dict['camera_head_lookatpoint'] = lookatpoint_obs_transformed[:, :3]
            if 'camera_head_lookatpoint' in self.action_indexing_raw:
                lookatpoint_action = data['action'][..., self.action_indexing_raw['camera_head_lookatpoint'][0]:self.action_indexing_raw['camera_head_lookatpoint'][1]]  # shape: (action_horizon, 3)
                lookatpoint_action_homogeneous = np.concatenate([
                    lookatpoint_action,
                    np.ones((lookatpoint_action.shape[0], 1))
                ], axis=-1)  # shape: (action_horizon, 4)
                ref_head_pose_action_mat = convert_pose_mat_rep(
                    pose_to_mat(np.concatenate([
                        data['action_raw']['gripper_head_eef_pos'],
                        data['action_raw']['gripper_head_eef_rot_axis_angle'],
                    ], axis=-1)),
                    base_pose_mat=base_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                # transform lookatpoint action as well
                lookatpoint_action_transformed = np.einsum('tij,tj->ti', ref_head_pose_action_mat, lookatpoint_action_homogeneous)  # (action_horizon, 4)
                actions[:, self.action_indexing['camera_head_lookatpoint'][0]:self.action_indexing['camera_head_lookatpoint'][1]] = lookatpoint_action_transformed[:, :3]
            
            # now transform all eef poses into the base frame
            for robot_id in range(self.num_robot):
                if f'{self.id_to_side[robot_id]}_eef_pos' in obs_dict and f'{self.id_to_side[robot_id]}_eef_rot_axis_angle' in obs_dict:
                    pose_mat = pose_to_mat(np.concatenate([
                        obs_dict[f'{self.id_to_side[robot_id]}_eef_pos'],
                        obs_dict[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle']
                    ], axis=-1))
                    ref_obs_pose_mat = convert_pose_mat_rep(
                        pose_mat, 
                        base_pose_mat=base_pose_mat[-1],
                        pose_rep='relative',
                        backward=False)
                    obs_pose = mat_to_pose10d(ref_obs_pose_mat)
                    obs_dict[f'{self.id_to_side[robot_id]}_eef_pos'] = obs_pose[:,:3]
                    obs_dict[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'] = obs_pose[:,3:]

                if f'{self.id_to_side[robot_id]}_eef_pos' in self.action_indexing_raw and f'{self.id_to_side[robot_id]}_eef_rot_axis_angle' in self.action_indexing_raw:
                    action_mat = pose_to_mat(np.concatenate([data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_pos'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_pos'][1]],
                                                            data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][1]],
                                                            ], axis=-1))
                    ref_action_pose_mat = convert_pose_mat_rep(
                        action_mat, 
                        base_pose_mat=base_pose_mat[-1],
                        pose_rep='relative',
                        backward=False)
                    action_pose = mat_to_pose10d(ref_action_pose_mat)
                    actions[:, self.action_indexing[f'{self.id_to_side[robot_id]}_eef_pos'][0]: self.action_indexing[f'{self.id_to_side[robot_id]}_eef_pos'][1]] = action_pose[:,:3]
                    actions[:, self.action_indexing[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][0]: self.action_indexing[f'{self.id_to_side[robot_id]}_eef_rot_axis_angle'][1]] = action_pose[:,3:]
                # gripper width
                if self.id_to_side[robot_id]+ '_gripper_width' in self.action_indexing_raw:
                    actions[:, self.action_indexing[f'{self.id_to_side[robot_id]}_gripper_width'][0]: self.action_indexing[f'{self.id_to_side[robot_id]}_gripper_width'][1]] = data['action'][..., self.action_indexing_raw[f'{self.id_to_side[robot_id]}_gripper_width'][0]: self.action_indexing_raw[f'{self.id_to_side[robot_id]}_gripper_width'][1]]    
            data['action'] = actions
        # additional metadata for debugging
        metadata = {
            'episode_idx': torch.tensor(data['metadata']['episode_idx'], dtype=torch.int64)
        }

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32)),
            'metadata': metadata
        }

        return torch_data
