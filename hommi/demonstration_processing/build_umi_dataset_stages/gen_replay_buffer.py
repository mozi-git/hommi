import sys
import os
from pathlib import Path
import zarr
import pickle
import yaml
import numpy as np
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
from hommi.common.cv_util import get_image_transform_with_border, depth2xyzmap
from hommi.demonstration_processing.utils.depth_util import load_depth
from hommi.common.replay_buffer import ReplayBuffer
from hommi.demonstration_processing.utils.lookat_util import (
    compute_center_lookat_from_pointmap,
)
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import numcodecs
register_codecs()

def gen_replay_buffer(session_dir: str, dataset_plan_path: str, out_replay_buffer_path: str, cfg: DictConfig):
    if cfg.num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = cfg.num_workers

    episode_keep_ratio = float(cfg.get("episode_keep_ratio", 1.0))
    if not (0.0 < episode_keep_ratio <= 1.0):
        raise ValueError(f"episode_keep_ratio must be in (0, 1], got {episode_keep_ratio}")
    episode_keep_ratio_name_filter = cfg.get("episode_keep_ratio_name_filter", None)
    if isinstance(episode_keep_ratio_name_filter, str) and not episode_keep_ratio_name_filter:
        episode_keep_ratio_name_filter = None
    if isinstance(episode_keep_ratio_name_filter, ListConfig):
        episode_keep_ratio_name_filter = list(episode_keep_ratio_name_filter)
    if episode_keep_ratio_name_filter is not None and not isinstance(episode_keep_ratio_name_filter, (str, list)):
        raise ValueError(
            "episode_keep_ratio_name_filter must be a string, list of strings, or null."
        )
    if isinstance(episode_keep_ratio_name_filter, list):
        episode_keep_ratio_name_filter = [str(token) for token in episode_keep_ratio_name_filter]

    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    n_grippers = -1
    n_cameras = -1
    buffer_start = 0
    vid_args = []
    demos_path = Path(session_dir).joinpath('demos')

    with open(dataset_plan_path, 'rb') as f:
        plan = pickle.load(f)

    for plan_episode in plan:
        grippers_by_side = {k.split('grippers_')[1]: v for k, v in plan_episode.items() if k.startswith('grippers_')}
        cameras_by_side = {k.split('cameras_')[1]: v for k, v in plan_episode.items() if k.startswith('cameras_')}

        # Validate side consistency
        if 'expected_sides' not in locals():
            expected_sides = {
                'grippers': sorted(grippers_by_side.keys()),
                'cameras': sorted(cameras_by_side.keys())
            }
        else:
            assert sorted(grippers_by_side.keys()) == expected_sides['grippers'], \
                f"Inconsistent gripper sides in episode {plan_episode['episode_name']}"
            assert sorted(cameras_by_side.keys()) == expected_sides['cameras'], \
                f"Inconsistent camera sides in episode {plan_episode['episode_name']}"

        if n_grippers == -1:
            n_grippers = sum(len(v) for v in grippers_by_side.values())
        else:
            assert n_grippers == sum(len(v) for v in grippers_by_side.values())

        if n_cameras == -1:
            n_cameras = sum(len(v) for v in cameras_by_side.values())
        else:
            assert n_cameras == sum(len(v) for v in cameras_by_side.values())

        episode_data = dict()
        # upsample_indexing_values = dict()
        # upsample_indexing_lengths = dict()

        n_frames_full = grippers_by_side['head'][0]['tcp_pose'].shape[0]
        episode_name = plan_episode['episode_name']
        use_ratio = episode_keep_ratio
        if episode_keep_ratio_name_filter is not None:
            if isinstance(episode_keep_ratio_name_filter, str):
                match = episode_keep_ratio_name_filter in episode_name
            else:
                match = any(token in episode_name for token in episode_keep_ratio_name_filter)
            if not match:
                use_ratio = 1.0
        n_frames = max(1, int(np.floor(n_frames_full * use_ratio)))
        if n_frames > n_frames_full:
            n_frames = n_frames_full
        if n_frames < n_frames_full:
            print(
                f"[gen_replay_buffer] Truncating episode {episode_name} "
                f"from {n_frames_full} to {n_frames} frames (ratio={use_ratio})"
            )

        # Grippers
        for side, grippers in grippers_by_side.items():
            prefix = f'gripper_{side}'
            gripper = grippers[0]  # Assuming only one gripper per side in the plan
            eef_pose = gripper['tcp_pose']
            eef_pos = eef_pose[..., :3][:n_frames]
            eef_rot = eef_pose[..., 3:][:n_frames]
            demo_start_pose = gripper['demo_start_pose'][:n_frames]
            demo_end_pose = gripper['demo_end_pose'][:n_frames]

            episode_data[f'{prefix}_eef_pos'] = eef_pos.astype(np.float32)
            episode_data[f'{prefix}_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
            episode_data[f'{prefix}_demo_start_pose'] = demo_start_pose.astype(np.float32)
            episode_data[f'{prefix}_demo_end_pose'] = demo_end_pose.astype(np.float32)

            if gripper['gripper_width'] is not None:
                gripper_widths = np.expand_dims(gripper['gripper_width'][:n_frames], axis=-1).astype(np.float32)
                episode_data[f'{prefix}_gripper_width'] = gripper_widths

        # Cameras
        for side, cameras in cameras_by_side.items():
            cam_key = f'camera_{side}'
            camera = cameras[0]  # Assuming only one camera per side in the plan
            # upsample_indexing_values[f'{cam_key}_ultrawide_rgb'] = camera['main_idx_to_ultrawide_idx']
            # start, end = camera['ultrawide_video_start_end']
            # upsample_indexing_lengths[f'{cam_key}_ultrawide_rgb'] = end - start

            main_video_path = demos_path.joinpath(camera['main_video_path']).absolute()
            ultrawide_video_path = demos_path.joinpath(camera['ultrawide_video_path']).absolute()
            depth_video_path = demos_path.joinpath(camera['depth_video_path']).absolute()
            try:
                assert main_video_path.is_file()
                assert ultrawide_video_path.is_file()
            except:
                print(f"Warning: Video files not found for {demos_path.joinpath(camera['main_video_path']).absolute()}.")
                exit(0)
            # assert depth_video_path.is_file()

            # main_start, main_end = camera['main_video_start_end']
            # ultra_start, ultra_end = camera['ultrawide_video_start_end']

            # if n_frames_main == -1:
            #     n_frames_main = main_end - main_start
            # else:
            #     assert n_frames_main == (main_end - main_start)

            # if n_frames_ultrawide == -1:
            #     n_frames_ultrawide = ultra_end - ultra_start
            # else:
            #     assert abs(n_frames_ultrawide - (ultra_end - ultra_start)) <= 1

            pose_idx_to_main_idx = camera['pose_idx_to_main_idx'][:n_frames]
            pose_idx_to_ultrawide_idx = camera['pose_idx_to_ultrawide_idx'][:n_frames]
            pose_idx_to_depth_idx = camera['pose_idx_to_depth_idx'][:n_frames]
            pose_count = len(pose_idx_to_main_idx)
            print(f"[gen_dataset_plan] episode {plan_episode['episode_name']} camera {cam_key} buffer_start={buffer_start} poses={pose_count}")
            vid_args.extend([
                {
                    'video_path': str(main_video_path),
                    'camera_key': cam_key,
                    'buffer_start': buffer_start,
                    'type': 'main_rgb',
                    'pose_idx_to_vid_idx': pose_idx_to_main_idx,
                },
                {
                    'video_path': str(ultrawide_video_path),
                    'camera_key': cam_key,
                    'buffer_start': buffer_start,
                    'type': 'ultrawide_rgb',
                    'pose_idx_to_vid_idx': pose_idx_to_ultrawide_idx,
                },
                {
                    'video_path': str(depth_video_path),
                    'camera_key': cam_key,
                    'buffer_start': buffer_start,
                    'type': 'depth',
                    'pose_idx_to_vid_idx': pose_idx_to_depth_idx,
                }])
            if cfg.save_pointmap and side == cfg.save_pointmap_side:
                vid_args.append({
                    'video_path': str(depth_video_path),
                    'camera_key': cam_key,
                    'buffer_start': buffer_start,
                    'type': 'pointmap',
                    'pose_idx_to_vid_idx': pose_idx_to_depth_idx,
                })

        buffer_start += n_frames

        out_replay_buffer.add_episode(
            data=episode_data,
            tasks=plan_episode['tasks'],
            compressors=None,
            episode_name=plan_episode['episode_name'],
            # upsample_indexing_values=upsample_indexing_values,
            # upsample_indexing_lengths=upsample_indexing_lengths
        )

    # Add videos to replay buffer
    if cfg.out_res_rgb is not None:
        out_res_rgb = (cfg.out_res_rgb, cfg.out_res_rgb)
    else:
        out_res_rgb = (224, 224)
    if cfg.out_res_pointmap is not None:
        out_res_pointmap = (cfg.out_res_pointmap, cfg.out_res_pointmap) 
    else:
        out_res_pointmap = (512, 512)
    num_episodes = len(plan)
    print(f"{num_episodes} episodes used in total ({len(vid_args)} videos)!")

    with av.open(vid_args[0]['video_path']) as container:
        main_ih, main_iw = container.streams.video[0].height, container.streams.video[0].width
    with av.open(vid_args[1]['video_path']) as container:
        ultrawide_ih, ultrawide_iw = container.streams.video[0].height, container.streams.video[0].width
    # with av.open(vid_args[2]['video_path']) as container:
    #     depth_ih, depth_iw = container.streams.video[0].height, container.streams.video[0].width
    depth_ih = 192
    depth_iw = 256

    if cfg.save_pointmap:
        # load depth intrinsic from config file (iphone_calibration: 'calibration/iphone15pro_calibration.yaml')
        depth_intrinsic = np.array(yaml.safe_load(open(os.path.abspath(cfg.iphone_calibration), 'r'))['depth']['intrinsicMatrix'])

    img_compressor = JpegXl(level=99, numthreads=1)
    float_compressor = numcodecs.Blosc(
        cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    for vid in vid_args:
        name = f"{vid['camera_key']}_{vid['type']}"
        is_depth_or_pointmap = vid['type'] in ('depth', 'pointmap')

        shape = (buffer_start,) + (out_res_pointmap if (is_depth_or_pointmap or vid['camera_key'].split('_')[-1] == cfg.save_pointmap_side) else out_res_rgb) + (1 if vid['type'] == 'depth' else 3,)
        dtype = np.float16 if is_depth_or_pointmap else np.uint8
        compressor = float_compressor if is_depth_or_pointmap else img_compressor

        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=shape,
            chunks=(1,) + (out_res_pointmap if (is_depth_or_pointmap or vid['camera_key'].split('_')[-1] == cfg.save_pointmap_side) else out_res_rgb) + (1 if vid['type'] == 'depth' else 3,),
            compressor=compressor,
            dtype=dtype
        )
        if vid['type'] == 'pointmap' and cfg.save_look_at_point:
            _ = out_replay_buffer.data.require_dataset(
                name=f"{vid['camera_key']}_lookatpoint",
                shape=(buffer_start,) + (3,),
                chunks=(1,) + (3,),
                compressor=compressor,
                dtype=dtype,
            )

    def video_to_zarr(replay_buffer, vid_metadata):
        vid_type = vid_metadata['type']
        if vid_type == 'pointmap':
            return  # Pointmap is handled separately
        cam_key = vid_metadata['camera_key']
        name = f"{cam_key}_{vid_type}"
        img_array = replay_buffer.data[name]
        pose_to_vid_idx = vid_metadata['pose_idx_to_vid_idx']
        buffer_offset = vid_metadata['buffer_start']
        if cfg.save_pointmap and vid_type == 'depth':
            pointmap_name = f"{cam_key}_pointmap"
            pointmap_array = replay_buffer.data[pointmap_name]
            if cfg.save_look_at_point:
                lookatpoint_name = f"{cam_key}_lookatpoint"
                lookatpoint_array = replay_buffer.data[lookatpoint_name]

        # Video resolution
        if vid_type == 'main_rgb':
            iw, ih = main_iw, main_ih
        elif vid_type == 'ultrawide_rgb':
            iw, ih = ultrawide_iw, ultrawide_ih
        elif vid_type == 'depth':
            iw, ih = depth_iw, depth_ih
        else:
            raise ValueError(f"Unknown video type: {vid_type}")

        if vid_type == 'depth':
            resize_tf_depth = get_image_transform_with_border(in_res=(iw, ih), out_res=out_res_pointmap, mode='depth')
            resize_tf_pointmap = get_image_transform_with_border(in_res=(iw, ih), out_res=out_res_pointmap, mode='pointmap')
            depth_data = load_depth(vid_metadata['video_path'], depth_shape=(depth_ih, depth_iw), dtype=np.float16)
            # confidence_path = vid_metadata['video_path'].replace('.raw', 'confidence.mp4')
            # with av.open(confidence_path) as container:
            #     in_stream = container.streams.video[0]
            #     in_stream.thread_count = 1
            #     decoded_frames = list(container.decode(in_stream))
            invalid_before = 0
            invalid_after = 0
            invalid_samples = []
            n_depth_frames = len(depth_data)
            for i, vid_frame_idx in enumerate(pose_to_vid_idx):
                if vid_frame_idx < 0:
                    invalid_before += 1
                    if len(invalid_samples) < 5:
                        invalid_samples.append((i, vid_frame_idx, 'before'))
                    continue
                if vid_frame_idx >= n_depth_frames:
                    invalid_after += 1
                    if len(invalid_samples) < 5:
                        invalid_samples.append((i, vid_frame_idx, 'after'))
                    continue
                if 0 <= vid_frame_idx < len(depth_data):
                    depth = depth_data[vid_frame_idx].copy()
                    # conf = decoded_frames[vid_frame_idx].to_ndarray(format='rgb24')[:, :, 0]
                    # depth[conf != 2] = 0  # Set invalid depth to 0
                    # pointmap = depth2xyzmap(depth, depth_intrinsic)
                    # os.makedirs('./runs', exist_ok=True)
                    # pcd_o3d = o3d.geometry.PointCloud()
                    # pcd_o3d.points = o3d.utility.Vector3dVector(pointmap.reshape(-1, 3))
                    # pcd_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(pointmap.reshape(-1, 3)))
                    # ply_path = os.path.join('./runs', f"pcd_{cam_key}_{vid_frame_idx}.ply")
                    # o3d.io.write_point_cloud(ply_path, pcd_o3d)
                    # print(f"Saved point cloud with attention colors to {ply_path}")
                    if cfg.save_depth:
                        depth_resized = resize_tf_depth(depth)
                        img_array[buffer_offset + i] = depth_resized[..., np.newaxis]
                    if cfg.save_pointmap:
                        pointmap = depth2xyzmap(depth, depth_intrinsic)
                        pointmap_resized = resize_tf_pointmap(pointmap)
                        pointmap_array[buffer_offset + i] = pointmap_resized
                        # print(f"[pointmap] {cam_key} frame {i} yielded pointmap (depth idx {vid_frame_idx}, depth min {np.nanmin(pointmap_resized)}, max {np.nanmax(pointmap_resized)})")
                        # import open3d as o3d
                        # os.makedirs('./runs', exist_ok=True)
                        # pcd_o3d = o3d.geometry.PointCloud()
                        # pcd_o3d.points = o3d.utility.Vector3dVector(pointmap_resized.reshape(-1, 3))
                        # pcd_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(pointmap_resized.reshape(-1, 3)))
                        # ply_path = os.path.join('./runs', f"pcd_{cam_key}_{vid_frame_idx}_resized.ply")
                        # o3d.io.write_point_cloud(ply_path, pcd_o3d)
                        # print(f"Saved point cloud with attention colors to {ply_path}")
                        if cfg.save_look_at_point:
                            look_at_point = compute_center_lookat_from_pointmap(
                                pointmap_resized
                            )
                            lookatpoint_array[buffer_offset + i] = look_at_point
            if invalid_before or invalid_after:
                print(
                    f"[gen_replay_buffer] Depth stream '{name}' skipped "
                    f"{invalid_before} frames before start and {invalid_after} after end "
                    f"(depth frames: {n_depth_frames}). Examples: {invalid_samples}"
                )
        else:
            resize_tf = get_image_transform_with_border(in_res=(iw, ih), out_res=(out_res_pointmap if vid_metadata['camera_key'].split('_')[-1] == cfg.save_pointmap_side else out_res_rgb), mode='rgb')
            with av.open(vid_metadata['video_path']) as container:
                in_stream = container.streams.video[0]
                in_stream.thread_count = 1
                decoded_frames = list(container.decode(in_stream))

                for i, vid_frame_idx in enumerate(pose_to_vid_idx):
                    if 0 <= vid_frame_idx < len(decoded_frames):
                        frame = decoded_frames[vid_frame_idx]
                        img = frame.to_ndarray(format='rgb24')
                        img = resize_tf(img)
                        img_array[buffer_offset + i] = img

    # for vid_metadata in vid_args:
    #     video_to_zarr(out_replay_buffer, vid_metadata)
    #     # save a sample pointmap for debugging
    #     if vid_metadata['type'] == 'pointmap':
    #         cam_key = vid_metadata['camera_key']
    #         pointmap_name = f"{cam_key}_pointmap"
    #         pointmap_array = out_replay_buffer.data[pointmap_name]
    #         if pointmap_array.shape[0] > 0:
    #             pointmap_sample = pointmap_array[0, ...].reshape(-1, 3)
    #             import open3d as o3d
    #             pcd_o3d = o3d.geometry.PointCloud()
    #             pcd_o3d.points = o3d.utility.Vector3dVector(pointmap_sample)
    #             pcd_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(pointmap_sample))
    #             ply_path = os.path.join('./runs', f"pcd_{cam_key}_sample.ply")
    #             o3d.io.write_point_cloud(ply_path, pcd_o3d)
    #             print(f"Saved sample point cloud to {ply_path}")
    #             exit(0)

    with tqdm(total=len(vid_args)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for vid_metadata in vid_args:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))
                futures.add(executor.submit(video_to_zarr, out_replay_buffer, vid_metadata))
            completed, _ = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print(f"Saving ReplayBuffer to {out_replay_buffer_path}")
    # Debug: inspect first few indices before writing to disk
    inspect_indices = []
    pointmap_key = 'camera_head_pointmap'
    if pointmap_key in out_replay_buffer.data:
        pointmap_arr = out_replay_buffer.data[pointmap_key]
        inspect_indices = list(range(min(8, pointmap_arr.shape[0])))
        for idx in inspect_indices:
            nz = int(np.count_nonzero(pointmap_arr[idx]))
            print(f"[debug] pointmap[{idx}] nonzero={nz}")
    depth_key = 'camera_head_depth'
    if depth_key in out_replay_buffer.data:
        depth_arr = out_replay_buffer.data[depth_key]
        if not inspect_indices:
            inspect_indices = list(range(min(8, depth_arr.shape[0])))
        for idx in inspect_indices:
            nz = int(np.count_nonzero(depth_arr[idx]))
            print(f"[debug] depth[{idx}] nonzero={nz}")
    with zarr.ZipStore(out_replay_buffer_path, mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    print(f"Done! {num_episodes} episodes used in total!")
