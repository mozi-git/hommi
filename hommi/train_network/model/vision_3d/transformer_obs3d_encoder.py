import os
import copy
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import timm
import logging

from hommi.train_network.model.vision_3d.pointcloud_base import PointCloudBaseEncoder
from hommi.train_network.model.vision_3d.position_encodings import NeRFSinusoidalPosEmb
from hommi.train_network.model.vision_3d.pointnet_extractor import AttentionExtractor
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from hommi.train_network.common.augmentation import ImageAugmentation

logger = logging.getLogger(__name__)

class TransformerObs3DEncoder(ModuleAttrMixin, PointCloudBaseEncoder):
    def __init__(
        self,
        shape_meta: dict,
        model_name: str='vit_base_patch16_clip_224.openai',
        global_pool: str='',
        train_image_transforms: Optional[ImageAugmentation]=None,
        eval_image_transforms: Optional[ImageAugmentation]=None,
        n_emb: int = 768,
        pretrained: bool=False,
        frozen: bool = False,
        use_group_norm: bool=False,
        share_rgb_model: bool=False,
        feature_aggregation: str=None,
        use_vision_norm: bool=True,
        concat_time_dimension: bool=True,

        pointcloud_model_name: str='vit_base_patch14_reg4_dinov2.lvd142m',
        freeze_pointcloud_model: bool=False,
        hidden_dim: int = 768,
        position_embedding_dim: int = 60,
        xyz_proj_type: str = "nerf",
        num_points: int = 512,
        downsample_mode: str = "feat",  # 'pos' for farthest point sampling, 'feat' for feature-based downsampling
        visualize_attention: bool=False,
        max_vis_frames: int = 4,
        do_crop: bool=True,
        **kwargs,
    ) -> None:
        # Initialize parent classes
        ModuleAttrMixin.__init__(self)
        PointCloudBaseEncoder.__init__(self,
            num_points=num_points,
            downsample_mode=downsample_mode,
            do_crop=do_crop,
            **kwargs
        )

        # Store shape metadata and configuration
        self.shape_meta = shape_meta
        self.model_name = model_name
        self.xyz_proj_type = xyz_proj_type
        self.hidden_dim = hidden_dim
        self.position_embedding_dim = position_embedding_dim
        self.n_emb = n_emb
        self.max_vis_frames = max_vis_frames
        self.train_image_transforms = train_image_transforms
        self.eval_image_transforms = eval_image_transforms

        rgb_keys = list()
        low_dim_keys = list()
        pointmap_keys = list()
        key_model_map = nn.ModuleDict()
        key_train_transform_map = nn.ModuleDict()
        key_eval_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        model, model_normalization_transform = self._init_backbone(
            model_name=model_name,
            global_pool=global_pool,
            pretrained=pretrained,
            frozen=frozen,
            use_group_norm=use_group_norm
        )

        pc_model, pc_model_normalization_transform = self._init_backbone(
            model_name=pointcloud_model_name,
            global_pool='',  # No global pooling for point cloud model
            pretrained=pretrained,
            frozen=freeze_pointcloud_model,
            use_group_norm=use_group_norm
        )

        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation is None:
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'

        obs_shape_meta = shape_meta['obs']

        for key, attr in obs_shape_meta.items():
            if obs_shape_meta[key].get('ignore_by_policy', False):
                continue
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                key_train_transform_map[key] = train_image_transforms.get_transform(key) if train_image_transforms is not None else nn.Identity()
                key_eval_transform_map[key] = eval_image_transforms.get_transform(key) if eval_image_transforms is not None else nn.Identity()

                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
            elif type == 'low_dim':
                dim = np.prod(shape)
                proj = nn.Identity()
                if dim != n_emb:
                    proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj

                low_dim_keys.append(key)
            elif type == 'pointmap':
                pointmap_keys.append(key)
                key_model_map[key] = pc_model
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = pc_model(example_img)
                    feature_shape = example_feature_map.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                if feature_size != self.hidden_dim:
                    proj = nn.Linear(in_features=feature_size, out_features=self.hidden_dim)
                key_projection_map[key] = proj
                # Calculate point cloud input dimension
                pc_in = (
                    self.hidden_dim
                    + (3 if self.xyz_proj_type == "none" else 2*self.position_embedding_dim)
                )
                pc_projection_map = AttentionExtractor(
                    in_shape=pc_in,
                    out_channels=n_emb,
                    use_layernorm=False,
                    final_norm='none',
                    hidden_dim=256,
                    num_heads=4,
                )
                self.pc_projection_map = pc_projection_map
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_train_transform_map = key_train_transform_map
        self.key_eval_transform_map = key_eval_transform_map
        self.model_normalization_transform = model_normalization_transform
        self.pc_model_normalization_transform = pc_model_normalization_transform
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.pointmap_keys = pointmap_keys
        self.key_shape_map = key_shape_map
        self.use_vision_norm = use_vision_norm
        self.concat_time_dimension = concat_time_dimension
        self.vis_attention = visualize_attention
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self._init_position_encoding()

    def _init_backbone(self, model_name: str, global_pool: str, pretrained: bool, frozen: bool, use_group_norm: bool):
        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )

        model_data_config = timm.data.resolve_data_config(model.pretrained_cfg)
        model_normalization_transform = torchvision.transforms.Normalize(
            mean=model_data_config['mean'],
            std=model_data_config['std']
        )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )
        return model, model_normalization_transform

    def aggregate_feature(self, feature):
        # Return: B, N, C

        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]

            # or use all tokens
            assert self.feature_aggregation is None
            return feature

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature

    def _init_position_encoding(self) -> None:
        """Initialize position encoding."""
        if self.xyz_proj_type == "nerf":
            self.xyz_proj = NeRFSinusoidalPosEmb(self.position_embedding_dim)
        elif self.xyz_proj_type == "none":
            self.xyz_proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported xyz_proj_type: {self.xyz_proj_type}")


    def forward(self, obs_dict):
        """Forward pass

        Args:
            obs_dict: Dictionary of observations with keys matching shape_meta

        Returns:
            embeddings: Tensor of shape [B, T_total, n_emb] where T_total includes
                       both point cloud and low-dim embeddings across time
        """
        batch_size = next(iter(obs_dict.values())).shape[0]
        embeddings = []

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])

            if self.training:
                img = self.key_train_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)

            if self.use_vision_norm:
                img = self.model_normalization_transform(img) # apply image normalization

            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B,-1,self.n_emb)
            embeddings.append(emb)

        # Process RGB + xyz pointmap inputs
        if self.pointmap_keys:
            rgb_data = []
            pcd_data = []

            for pcd_key in self.pointmap_keys:
                rgb = obs_dict[pcd_key.replace('pointmap', 'main_rgb')] # B, T, C, H, W
                pcd = obs_dict[pcd_key] # B, T, C, H, W
                assert rgb.shape == pcd.shape
                rgb_key = pcd_key.replace('pointmap', 'main_rgb')
                if (
                    (self.training and rgb_key in self.key_train_transform_map)
                    or (not self.training and rgb_key in self.key_eval_transform_map)
                ):
                    B, T = rgb.shape[:2]
                    rgb_flat = rgb.reshape(B * T, *rgb.shape[2:])
                    pcd_flat = pcd.reshape(B * T, *pcd.shape[2:])
                    if self.training and self.train_image_transforms is not None:
                        rgb_flat, pcd_flat = self.train_image_transforms.apply_shared(
                            rgb_key, rgb_flat, pcd_flat
                        )
                    elif (not self.training) and self.eval_image_transforms is not None:
                        rgb_flat, pcd_flat = self.eval_image_transforms.apply_shared(
                            rgb_key, rgb_flat, pcd_flat
                        )
                    else:
                        transform = (
                            self.key_train_transform_map[rgb_key]
                            if self.training
                            else self.key_eval_transform_map[rgb_key]
                        )
                        rgb_flat = transform(rgb_flat)
                    rgb = rgb_flat.reshape(B, T, *rgb_flat.shape[1:])
                    pcd = pcd_flat.reshape(B, T, *pcd_flat.shape[1:])
                rgb_data.append(rgb)
                pcd_data.append(pcd)

            rgb = torch.stack(rgb_data, dim=0)  # n_cam, B, T, C, H, W
            pcd = torch.stack(pcd_data, dim=0)  # n_cam, B, T, C, H, W

            # save pointcloud here to debug
            # print(f"pcd shape: {pcd.shape}, rgb shape: {rgb.shape}")
            # save_attention_data(pcd[0, 0, 0, :3, ...].reshape(3, -1).T.cpu().numpy(), np.ones_like(pcd[0, 0, 0, :3, ...].reshape(3, -1).T.cpu().numpy()), './runs/attention_visualization', 'input_debug')

            n_cam, B, T = rgb.shape[:3]

            rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
            pcd = einops.rearrange(pcd, "ncam b fs c h w -> (b fs ncam) c h w")

            # Process through backbone
            rgb_normalized = self.pc_model_normalization_transform(rgb)    # B*T*n_cam, C, H, W
            rgb_features = self.key_model_map[pcd_key](rgb_normalized)      # B*T*n_cam, h*w+1, feature_dim
            rgb_features = self.key_projection_map[pcd_key](rgb_features)  # B*T*n_cam, h*w+1, hidden_dim
            rgb_features = rgb_features[:, 1:, :]  # Remove the first token (CLS token)
            # rgb_features.shape[1] should be a square number h*w, so we can rearrange it
            assert int(np.sqrt(rgb_features.shape[1]))**2 == rgb_features.shape[1], \
                f"Expected rgb_features shape[1] to be a perfect square, got {rgb_features.shape[1]}"

            feat_h, feat_w = int(np.sqrt(rgb_features.shape[1])), int(np.sqrt(rgb_features.shape[1]))
            pcd = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")  # B*T*n_cam, C, feat_h, feat_w

            pcd = einops.rearrange(
                pcd, "(b fs ncam) c h w -> b fs (ncam h w) c", ncam=n_cam, fs=T
            )
            rgb_features = einops.rearrange(
                rgb_features, "(b fs ncam) (h w) c -> b fs (ncam h w) c", b=B, ncam=n_cam, fs=T, h=feat_h, w=feat_w
            )
#
            # save_attention_data(pcd[0, :3, ...].reshape(-1,3).cpu().numpy(), np.ones_like(pcd[0, :3, ...].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_before_mask_debug')

            mask = pcd[..., -1] > 0.1
            # print(f"mask shape: {mask.shape}, masked points: {torch.sum(~mask)} out of {mask.numel()}")
            mask_hand, pcd_in_left, pcd_in_right = self._crop_point_cloud_with_hands(pcd, obs_dict['gripper_head_eef_pos_wrt_left'], obs_dict['gripper_head_eef_rot_axis_angle_wrt_left'], obs_dict['gripper_head_eef_pos_wrt_right'], obs_dict['gripper_head_eef_rot_axis_angle_wrt_right'])
            if self.do_crop:
                mask = torch.logical_and(mask, mask_hand)
            # print(f"mask shape after hand crop: {mask.shape}, masked points: {torch.sum(~mask)} out of {mask.numel()}")
            pcd = pcd * mask.unsqueeze(-1)  # Apply mask to point cloud
            pcd_in_left = pcd_in_left * mask.unsqueeze(-1)  # Apply mask to left hand point cloud
            pcd_in_right = pcd_in_right * mask.unsqueeze(-1)
            rgb_features = rgb_features * mask.unsqueeze(-1)  # Apply mask to RGB features

            # save_attention_data(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy(), np.ones_like(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_before_downsample_debug')

            # Downsample point cloud
            pcd, rgb_features, _, mask, pcd_in_left, pcd_in_right = self._downsample_point_cloud(
                pcd=pcd, rgb_features=rgb_features, mask=mask, num_points=self.num_points, pcd_left=pcd_in_left, pcd_right=pcd_in_right
            )
            # print(f"pcd shape after downsample: {pcd.shape}, rgb_features shape: {rgb_features.shape}, mask shape: {mask.shape}")
            # print(f"masked points after downsample: {torch.sum(~mask)} out of {mask.numel()}")
            # pcd shape: B, T, n_points, C; rgb_features shape: B, T, n_points, hidden_dim; mask shape: B, T, n_points
            # save_attention_data(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy(), np.ones_like(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_after_downsample_debug')

            # Apply position encoding
            # pcd_pos_emb = self.xyz_proj(pcd)    # B, T, n_points, hidden_dim
            pcd_pos_emb_left = self.xyz_proj(pcd_in_left)    # B, T, n_points, pos_emb_dim
            pcd_pos_emb_right = self.xyz_proj(pcd_in_right)    # B, T, n_points, pos_emb_dim
            # cat_cloud = torch.cat([pcd_pos_emb, rgb_features], dim=-1)
            cat_cloud = torch.cat([pcd_pos_emb_left, pcd_pos_emb_right, rgb_features], dim=-1)
            if self.vis_attention:
                pc_emb, attn_weights = self.pc_projection_map(cat_cloud, mask=mask, return_attention=True)  # B, T, n_emb
                self.visualize_attention(attn_weights, pcd, pcd_data, rgb_data)
            else:
                pc_emb = self.pc_projection_map(cat_cloud, mask=mask)  # B, T, n_emb
            assert pc_emb.shape[-1] == self.n_emb, f"Expected pc_emb shape[-1] to be {self.n_emb}, got {pc_emb.shape[-1]}"
            embeddings.append(pc_emb)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)

        # concatenate all features along t
        if self.concat_time_dimension:
            result = torch.cat(embeddings, dim=1) # (B, T, n_emb)
        else:
            result = torch.stack(embeddings, dim=2) # (B, T, num_obs_types, n_emb)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        if self.concat_time_dimension:
            assert len(example_output.shape) == 3
        else:
            assert len(example_output.shape) == 4
        assert example_output.shape[0] == 1
        assert example_output.shape[-1] == self.n_emb

        return example_output.shape

    def visualize_attention(self, attn_weights, pcd, pcd_data, rgb_data):
        """ Visualize attention weights for point cloud data. Highlight the points with attention weights.

        Args:
            attn_weights: Attention weights from the model, shape [B, T, n_points]
            pcd: Point cloud data, shape [B, T, n_points, C]
            pcd_data: full point cloud before downsampling, list of n_cam, [B, T, C, H, W]
            rgb_data: RGB data corresponding to the point cloud, list of n_cam, [B, T, C, H, W]
        """
        from datetime import datetime
        from hommi.train_network.model.vision_3d.point_cloud_utils import show_point_cloud, show_point_cloud_plt, show_point_cloud_viser

        # Ensure tensors are moved to CPU and converted to numpy
        attn_weights = attn_weights.detach().cpu().numpy()
        pcd = pcd.detach().cpu().numpy()

        B, T, n_points, C = pcd.shape
        assert attn_weights.shape == (B, T, n_points), \
            f"Expected attn_weights shape to be (B, T, n_points), got {attn_weights.shape}"

        # Create output directory if it doesn't exist
        output_dir = './runs/attention_visualization'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        total_frames = B * T
        if total_frames == 0 or self.max_vis_frames <= 0:
            return

        num_samples = min(self.max_vis_frames, total_frames)
        flat_indices = np.random.choice(total_frames, size=num_samples, replace=False)

        for flat_idx in flat_indices:
            b = flat_idx // T
            t = flat_idx % T
            tag = f"{timestamp}_b{b}_t{t}"

            points_3d = pcd[b, t, :, :3]  # Extract XYZ coordinates
            weights = attn_weights[b, t]

            # Normalize attention weights to [0, 1] for color mapping
            weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

            # Create color map based on attention weights (viridis-like colors)
            colors = create_attention_colormap(weights_normalized)

            print(f"Attention weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

            # Method 1: Use Open3D visualization (interactive)
            # print("Showing interactive Open3D visualization...")
            show_point_cloud(
                pcd=points_3d,
                pcd_colors=colors,
                show_axes=True
            )

            # print("Creating matplotlib visualization...")
            # show_point_cloud_plt(
            #     pcd=points_3d,
            #     pcd_colors=colors
            # )

            save_attention_data(points_3d, colors, output_dir, tag)
            # save full attention map
            save_attention_data(
                pcd_data[0][b, t, :3, ...].reshape(3, -1).T.cpu().numpy(),
                rgb_data[0][b, t, :3, ...].reshape(3, -1).T.cpu().numpy(),
                output_dir,
                f"full_{tag}"
            )

            full_pcd = pcd_data[0][b, t, :3, ...].reshape(3, -1).T.cpu().numpy()
            full_colors = interpolate_colors_knn(
                source_points=points_3d,
                source_colors=colors,
                target_points=full_pcd,
                k=5,  # Use 5 nearest neighbors
                max_distance=None
            )
            print(f"Interpolated full point cloud colors shape: {full_colors.shape}")
            save_attention_data(full_pcd, full_colors, output_dir, f"full_interpolated_{tag}")

def interpolate_colors_knn(source_points, source_colors, target_points, k=3, max_distance=None):
    """
    Interpolate colors from source points to target points using k-nearest neighbors.

    Args:
        source_points: (N, 3) array of source point coordinates
        source_colors: (N, 3) array of source point colors
        target_points: (M, 3) array of target point coordinates
        k: number of nearest neighbors to use for interpolation
        max_distance: maximum distance for interpolation (optional)

    Returns:
        interpolated_colors: (M, 3) array of interpolated colors
    """
    from sklearn.neighbors import NearestNeighbors
    # Build KNN model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(source_points)
    distances, indices = nbrs.kneighbors(target_points)

    # Initialize output colors
    interpolated_colors = np.zeros((len(target_points), 3))

    for i in range(len(target_points)):
        neighbor_distances = distances[i]
        neighbor_indices = indices[i]

        # Filter out neighbors that are too far (if max_distance is specified)
        if max_distance is not None:
            valid_mask = neighbor_distances <= max_distance
            neighbor_distances = neighbor_distances[valid_mask]
            neighbor_indices = neighbor_indices[valid_mask]

        if len(neighbor_distances) == 0:
            # No valid neighbors, use default color or nearest anyway
            interpolated_colors[i] = source_colors[indices[i][0]]
            continue

        # Inverse distance weighting
        if neighbor_distances[0] < 1e-8:  # Very close point
            interpolated_colors[i] = source_colors[neighbor_indices[0]]
        else:
            weights = 1.0 / (neighbor_distances + 1e-8)
            weights = weights / weights.sum()
            interpolated_colors[i] = np.sum(source_colors[neighbor_indices] * weights[:, np.newaxis], axis=0)

    return interpolated_colors

def create_attention_colormap(weights_normalized):
    """Create viridis-like colormap for attention weights."""
    import matplotlib.cm as cm

    # Use matplotlib's viridis colormap
    viridis = cm.get_cmap('viridis')
    colors = viridis(weights_normalized)[:, :3]  # Remove alpha channel, if present

    return colors


def save_attention_data(points, colors, output_dir, timestamp):
    """Save point cloud with attention colors to a PLY file."""
    # points [N, 3], colors [N, 3]
    try:
        import open3d as o3d
        os.makedirs(output_dir, exist_ok=True)
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        ply_path = os.path.join(output_dir, f"pcd_attention_{timestamp}.ply")
        o3d.io.write_point_cloud(ply_path, pcd_o3d)
        print(f"Saved point cloud with attention colors to {ply_path}")
    except ImportError:
        print("Open3D not available, skipping PLY file save")
    except Exception as e:
        print(f"Failed to save PLY file: {e}")
