import einops

import dgl.geometry as dgl_geo
import torch

from hommi.train_network.model.vision_3d.base import BaseEncoder
import hommi.train_network.model.vision_3d.point_cloud_utils as pcu


class PointCloudBaseEncoder(BaseEncoder):
    def __init__(
        self,
        num_points=None,
        do_crop=True,
        do_hand_crop=True,
        downsample_mode="pos",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_points = num_points
        self.shape_meta = self.shape_meta
        self.do_crop = do_crop
        self.do_hand_crop = do_hand_crop
        self.downsample_mode = downsample_mode

        boundaries = torch.tensor(((-1, -1, -1), (1, 1, 1)))    # TODO: this is a placeholder, should be replaced with actual boundaries
        boundaries = einops.repeat(boundaries, "i j -> 1 i j")
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32))
        self.boundaries: torch.Tensor = self.boundaries # To help with type checking
        
        hand_frame_boundaries = torch.tensor(((-1, -1, 0), (1, 1, 1)), dtype=torch.float32)
        self.register_buffer('hand_frame_boundaries', hand_frame_boundaries)
        self.hand_frame_boundaries: torch.Tensor = self.hand_frame_boundaries # To help with type checking

        head_frame_boundaries = torch.tensor(((-1, -1, 0), (1, 1, 2)), dtype=torch.float32)
        self.register_buffer('head_frame_boundaries', head_frame_boundaries)
        self.head_frame_boundaries: torch.Tensor = self.head_frame_boundaries
        

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PointCloudBaseEncoder is an abstract class")

    def _downsample_point_cloud(self, pcd, mask=None, rgb=None, rgb_features=None, num_points=None, pcd_left=None, pcd_right=None):
        if self.downsample_mode == "none":
            # No downsampling - return inputs as is
            return pcd, rgb_features, rgb, mask
        
        B, fs, N, D = pcd.shape
        device = pcd.device
        # Reshape inputs for batch processing
        pcd = einops.rearrange(pcd, "b fs n d -> (b fs) n d")
        if mask is not None:
            mask = einops.rearrange(mask, "b fs n -> (b fs) n")
        if rgb is not None:
            rgb = einops.rearrange(rgb, "b fs n d -> (b fs) n d")
        if rgb_features is not None:
            rgb_features = einops.rearrange(rgb_features, "b fs n d -> (b fs) n d")
        if pcd_left is not None:
            pcd_left = einops.rearrange(pcd_left, "b fs n d -> (b fs) n d")
        if pcd_right is not None:
            pcd_right = einops.rearrange(pcd_right, "b fs n d -> (b fs) n d")

        num_points = num_points or self.num_points

        # Get indices for sampling points
        if self.downsample_mode == "pos":
            # Sample points with FPS based on point cloud positions
            downsample_indices = dgl_geo.farthest_point_sampler(pcd, num_points, 0)
            downsample_indices_clipped = torch.clamp(downsample_indices, min=0)

        elif self.downsample_mode == "feat":
            assert rgb_features is not None
            # Sample points with FPS if features available, otherwise uniform
            downsample_indices = dgl_geo.farthest_point_sampler(
                rgb_features[..., :30], num_points, 0
            )
            downsample_indices_clipped = downsample_indices

        # Gather points and features using indices
        downsampled_pcd = torch.gather(
            pcd, 1, einops.repeat(downsample_indices_clipped, "b n -> b n d", d=pcd.shape[-1])
        )
        downsampled_pcd = einops.rearrange(downsampled_pcd, "(b fs) n d -> b fs n d", b=B)

        if mask is not None:
            downsample_mask = torch.gather(mask, 1, downsample_indices_clipped)
            downsample_mask = einops.rearrange(downsample_mask, "(b fs) n -> b fs n", b=B)
        else:
            downsample_mask = None
        
        if rgb is not None:
            downsampled_rgb = torch.gather(
                rgb, 1, einops.repeat(downsample_indices_clipped, "b n -> b n 3")
            )
            downsampled_rgb = einops.rearrange(downsampled_rgb, "(b fs) n d -> b fs n d", b=B)
        else:
            downsampled_rgb = None
            
        if rgb_features is not None:
            downsampled_feats = torch.gather(
                rgb_features,
                1,
                einops.repeat(downsample_indices_clipped, "b n -> b n k", k=rgb_features.shape[-1]),
            )
            downsampled_feats = einops.rearrange(downsampled_feats, "(b fs) n d -> b fs n d", b=B)
        else:
            downsampled_feats = None

        if pcd_left is not None:
            downsampled_pcd_left = torch.gather(
                pcd_left, 1, einops.repeat(downsample_indices_clipped, "b n -> b n d", d=pcd_left.shape[-1])
            )
            downsampled_pcd_left = einops.rearrange(downsampled_pcd_left, "(b fs) n d -> b fs n d", b=B)
        else:
            downsampled_pcd_left = None

        if pcd_right is not None:
            downsampled_pcd_right = torch.gather(
                pcd_right, 1, einops.repeat(downsample_indices_clipped, "b n -> b n d", d=pcd_right.shape[-1])
            )
            downsampled_pcd_right = einops.rearrange(downsampled_pcd_right, "(b fs) n d -> b fs n d", b=B)
        else:
            downsampled_pcd_right = None


        return downsampled_pcd, downsampled_feats, downsampled_rgb, downsample_mask, downsampled_pcd_left, downsampled_pcd_right

    def _crop_point_cloud(self, pcd, task_id=None, hand_mat_inv=None, boundaries=None):
        """
        Apply cropping to point cloud data based on boundaries and hand frame.
        
        Args:
            pcd: Point cloud data
            rgb_features: RGB features
            rgb: RGB data
            data: Input data dictionary
            device: Target device
            
        Returns:
            Tuple of (cropped_pcd, cropped_rgb_features, cropped_rgb, mask)
        """
        device = pcd.device
        B, fs, _, _ = pcd.shape
        pcd = einops.rearrange(pcd, "b fs n d -> (b fs) n d")
        mask = torch.ones(pcd.shape[:-1], device=device).bool()
        
        if self.do_crop:
            if boundaries is None:
                boundaries = self.boundaries[task_id]
            boundaries = einops.repeat(boundaries, 'b n d -> (b fs) n d', fs=fs, b=B)
            mask = pcu.crop_point_cloud(pcd, boundaries)

        if self.do_hand_crop:
            assert hand_mat_inv is not None

            hand_mat_invs = hand_mat_inv if type(hand_mat_inv) == list else [hand_mat_inv]
            
            for hand_mat_inv in hand_mat_invs:
                hand_mat_inv = hand_mat_inv[:, -1] # Take last hand mat along frame stack dimension
                pcd_hand = pcu.batch_transform_point_cloud(pcd, hand_mat_inv)

                boundaries = einops.repeat(self.hand_frame_boundaries, 'n d -> (b fs) n d', b=B, fs=fs)
                hand_mask = pcu.crop_point_cloud(pcd_hand, boundaries)
                mask = torch.logical_and(mask, hand_mask)

        mask = einops.rearrange(mask, "(b fs) n -> b fs n", b=B)
        
        return mask

    def _crop_point_cloud_with_hands(self, pcd, left_eef_pos_inv, left_eef_rot_inv, right_eef_pos_inv, right_eef_rot_inv):
        """
        Crop point cloud data based on hand positions and orientations.

        Args:
            pcd: Point cloud data, [B, T, N, D]
            left_eef_pos_inv: head wrt Left end-effector position [B, T, 3]
            left_eef_rot_inv: head wrt Left end-effector rotation [B, T, 6]
            right_eef_pos_inv: head wrt Right end-effector position [B, T, 3]
            right_eef_rot_inv: head wrt Right end-effector rotation [B, T, 6]

        Returns:
            mask: Boolean mask indicating which points are within the hand frames
        """
        left_eef_mat_inv = pcu.pose_to_mat(torch.concatenate([left_eef_pos_inv, left_eef_rot_inv], dim=-1))
        right_eef_mat_inv = pcu.pose_to_mat(torch.concatenate([right_eef_pos_inv, right_eef_rot_inv], dim=-1))
        # transform point cloud to hand frames
        pcd_in_left_hand_frame = pcu.batch_transform_point_cloud(pcd, left_eef_mat_inv)
        pcd_in_right_hand_frame = pcu.batch_transform_point_cloud(pcd, right_eef_mat_inv)
        left_mask = pcd_in_left_hand_frame[..., -1] > -0.13     # the length of finger
        right_mask = pcd_in_right_hand_frame[..., -1] > -0.13
        mask = torch.logical_and(left_mask, right_mask)
        return mask, pcd_in_left_hand_frame, pcd_in_right_hand_frame
