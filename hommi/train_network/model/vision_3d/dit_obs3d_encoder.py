import os
import copy
from typing import Optional
from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import timm
import logging
from transformers import AutoModel
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

from hommi.train_network.model.vision_3d.pointcloud_base import PointCloudBaseEncoder
from hommi.train_network.model.vision_3d.position_encodings import NeRFSinusoidalPosEmb
from hommi.train_network.model.vision_3d.pointnet_extractor import AttentionExtractor, MultiModalSelfAttention, WristHeadCrossAttention, MaskedImageTokenizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from hommi.train_network.common.augmentation import ImageAugmentation
from hommi.train_network.model.vision_3d.pointcloud_augmentation import PointCloud3DAugmentation

logger = logging.getLogger(__name__)

class DiTObs3DEncoder(ModuleAttrMixin, PointCloudBaseEncoder):
    def __init__(
        self,
        shape_meta: dict,
        model_name: str='vit_base_patch16_clip_224.openai',
        global_pool: str='',
        train_image_transforms: Optional[ImageAugmentation]=None,
        eval_image_transforms: Optional[ImageAugmentation]=None,
        pretrained: bool=False,
        frozen: bool = False,
        use_group_norm: bool=False,
        share_rgb_model: bool=False,
        feature_aggregation: str=None,
        use_vision_norm: bool=True,

        pointcloud_model_name: str='vit_base_patch14_dinov2.lvd142m',
        share_pointcloud_model: bool=False,
        freeze_pointcloud_model: bool=False,
        position_embedding_dim: int = 60,
        xyz_proj_type: str = "nerf",
        num_points: int = 512,
        downsample_mode: str = "feat",  # 'pos' for farthest point sampling, 'feat' for feature-based downsampling
        visualize_attention: bool=False,
        debug_nan_checks: bool=False,
        max_vis_frames: int = 4,
        # do_crop: bool=True,
        # egocentric_frame: bool=True,   # use pointcloud in egocentric frame
        add_pos_emb: bool=False,     # add positional embedding to feature instead of concatenating it
        wrist_pos_emb: bool=False,       # add wrist pos as positional embedding to the wrist rgb features

        pointcloud_augmentation: Optional[PointCloud3DAugmentation] = None,

        head_wrist_attention: str = 'none', # none, cls, patch
        n_emb: int = 768,
        wrist_mask_prob: float = 0.0,   # the probability to mask out wrist images during training
        **kwargs,
    ) -> None:
        # Initialize parent classes
        ModuleAttrMixin.__init__(self)
        PointCloudBaseEncoder.__init__(self,
            num_points=num_points,
            downsample_mode=downsample_mode,
            # do_crop=do_crop,
            **kwargs
        )

        # Store shape metadata and configuration
        self.shape_meta = shape_meta
        self.model_name = model_name
        self.xyz_proj_type = xyz_proj_type
        self.position_embedding_dim = position_embedding_dim
        self.head_wrist_attention = head_wrist_attention
        self.add_pos_emb = add_pos_emb
        self.wrist_pos_emb = wrist_pos_emb
        self.feature_aggregation = feature_aggregation

        obs_shape_meta = shape_meta['obs']
        has_policy_visible_rgb = any(
            (not attr.get('ignore_by_policy', False))
            and attr.get('type', 'low_dim') == 'rgb'
            for attr in obs_shape_meta.values()
        )
        if not has_policy_visible_rgb:
            logger.info(
                "No policy-visible RGB inputs were found; skipping wrist RGB backbone initialization."
            )
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

        feature_size = n_emb
        num_patches = 0
        if has_policy_visible_rgb:
            model, model_normalization_transform = self._init_backbone(
                model_name=model_name,
                global_pool=global_pool,
                pretrained=pretrained,
                frozen=frozen,
                use_group_norm=use_group_norm
            )
            with torch.no_grad():
                shape = (3, 224, 224) # default input shape
                for key, attr in obs_shape_meta.items():
                    if attr.get('ignore_by_policy', False):
                        continue
                    if attr.get('type', 'low_dim') == 'rgb':
                        shape = tuple(attr['shape'])
                        break
                example_img = torch.zeros((1,)+tuple(shape))
                example_feature_map = model(example_img)
                example_features = self.aggregate_feature(example_feature_map)
                feature_shape = example_features.shape
                feature_size = feature_shape[-1]
                num_patches = example_features.shape[1]
                logger.info(f"Wrist vit has {num_patches} patches")
        else:
            model = None
            model_normalization_transform = nn.Identity()

        effective_share_pointcloud_model = share_pointcloud_model and has_policy_visible_rgb
        if share_pointcloud_model and not has_policy_visible_rgb:
            logger.info("share_pointcloud_model ignored because RGB backbone is disabled.")

        if effective_share_pointcloud_model:
            assert model_name == pointcloud_model_name, f"To share pointcloud model, model_name ({model_name}) must be the same as pointcloud_model_name ({pointcloud_model_name})"
            assert isinstance(model, nn.Module)
            pc_model = model
            pc_model_normalization_transform = model_normalization_transform
        else:
            pc_model, pc_model_normalization_transform = self._init_backbone(
                model_name=pointcloud_model_name,
                global_pool='',  # No global pooling for point cloud model
                pretrained=pretrained,
                frozen=freeze_pointcloud_model,
                use_group_norm=use_group_norm
            )

        # handle feature aggregation
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation is None:
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'

        # handle sharing vision backbone
        if has_policy_visible_rgb and share_rgb_model:
            assert isinstance(model, nn.Module)
            key_model_map["rgb"] = model
            proj = nn.Identity()
            if feature_size != n_emb:
                proj = nn.Linear(in_features=feature_size, out_features=n_emb)
            key_projection_map["rgb"] = proj

        for key, attr in obs_shape_meta.items():
            if obs_shape_meta[key].get('ignore_by_policy', False):
                continue
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                if not has_policy_visible_rgb:
                    raise ValueError("RGB observation detected but RGB backbone is disabled.")
                rgb_keys.append(key)
                if not share_rgb_model:
                    this_model = copy.deepcopy(model)
                    key_model_map[key] = this_model
                    proj = nn.Identity()
                    if feature_size != n_emb:
                        proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                    key_projection_map[key] = proj
                key_train_transform_map[key] = train_image_transforms.get_transform(key) if train_image_transforms is not None else nn.Identity()
                key_eval_transform_map[key] = eval_image_transforms.get_transform(key) if eval_image_transforms is not None else nn.Identity()
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'pointmap':
                pointmap_keys.append(key)
                key_model_map[key] = pc_model
                hidden_dim = 768
                with torch.no_grad():
                    if self._is_dinov3_model(pointcloud_model_name):
                        # For DINOv3, use the processor and get hidden size from config
                        hidden_dim = pc_model.config.hidden_size
                    else:
                        example_img = torch.zeros((1,)+tuple(shape))
                        example_features = pc_model(example_img)
                        hidden_dim = example_features.shape[-1]
                # Calculate point cloud input dimension
                if self.add_pos_emb:
                    pc_in = hidden_dim
                    assert pc_in == n_emb
                    self.position_embedding_dim = n_emb 
                else:
                    pc_in = (
                        hidden_dim
                        + (3 if self.xyz_proj_type == "none" else self.position_embedding_dim)
                    )
                # pc_out = n_emb if (head_wrist_attention == 'cls' or head_wrist_attention == 'patch') else pc_in
                if head_wrist_attention == 'patch':
                    pc_projection_map = nn.Identity()
                else:
                    pc_projection_map = AttentionExtractor(
                        in_shape=pc_in,
                        out_channels=n_emb if (head_wrist_attention == 'cls') else pc_in,
                        use_layernorm=False,
                        final_norm='none',
                        hidden_dim=256,
                        num_heads=4,
                    )
                self.pc_projection_map = pc_projection_map
                # Head embeddings fed into the wrist/head attention module should
                # match the dimensionality of the wrist embeddings. When we use
                # CLS attention, the projection map above already outputs `n_emb`
                # features, so keep the recorded dimension consistent.
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        if len(rgb_keys) == 0 and head_wrist_attention != 'none':
            raise ValueError("head_wrist_attention requires RGB observations, but none were found in shape_meta.")

        # handle head-wrist attention
        if head_wrist_attention == 'cls':
            # use the CLS token feature from wrist images and aggregated 3d feature from head to do self-attention
            self.head_wrist_attention_module = MultiModalSelfAttention(
                n_emb=n_emb,
                num_heads=4,
                dropout=0.1
            )
        elif head_wrist_attention == 'patch':
            self.head_wrist_attention_module = WristHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                wrist_num_patches=num_patches
            )
        else:
            assert head_wrist_attention == 'none'
            self.head_wrist_attention_module = None

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_train_transform_map = key_train_transform_map
        self.key_eval_transform_map = key_eval_transform_map
        self.key_projection_map = key_projection_map
        self.model_normalization_transform = model_normalization_transform
        self.pointcloud_model_name = pointcloud_model_name
        self.pc_model_normalization_transform = pc_model_normalization_transform
        self.share_rgb_model = share_rgb_model
        self.share_pointcloud_model = effective_share_pointcloud_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.pointmap_keys = pointmap_keys
        self.key_shape_map = key_shape_map
        self.obs_normalizer = None
        self._pointmap_denorm_cache = {}
        self.attn_force_eager = False
        self.use_vision_norm = use_vision_norm
        self.vis_attention = visualize_attention
        self.debug_nan_checks = debug_nan_checks
        self.num_patches = num_patches
        if self._is_dinov3_model(model_name) and model is not None and hasattr(model, "config"):
            self.dinov3_num_register_tokens = model.config.num_register_tokens
        else:
            self.dinov3_num_register_tokens = 0
        self.attention_hooks_on = False
        self.pointcloud_augmentation = pointcloud_augmentation
        self.max_vis_frames = max_vis_frames
        self.wrist_mask_prob = wrist_mask_prob
        self._rng = np.random.default_rng(seed=0)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self._init_position_encoding()
        # if visualize_attention:
        #     self.setup_vit_attention_hooks()
            
        if self.vis_attention or self.debug_nan_checks:
            for name, module in self.named_modules():
                if isinstance(module, nn.Module) and len(list(module.children())) == 0:  # Only hook leaf modules
                    def hook_fn(module, input, output, layer_name=name):
                        # Handle different output types
                        if isinstance(output, torch.Tensor):
                            # Direct tensor output
                            # print(f"Layer {layer_name}:")
                            # print(f"  Output shape: {output.shape}")
                            # print(f"  Output dtype: {output.dtype}")
                            # print(f"  Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, "
                            #     f"mean={output.mean().item():.4f}, std={output.std().item():.4f}")

                            # Check for NaN/Inf
                            if torch.isnan(output).any():
                                print(f"  WARNING: NaN detected in output layer {layer_name}!")
                            if torch.isinf(output).any():
                                print(f"  WARNING: Inf detected in output layer {layer_name}!")

                        elif isinstance(output, tuple):
                            # Tuple output (like from rope_embeddings)
                            # print(f"Layer {layer_name}: (tuple with {len(output)} elements)")
                            for i, tensor in enumerate(output):
                                if isinstance(tensor, torch.Tensor):
                                    # print(f"  Element {i} - shape: {tensor.shape}, dtype: {tensor.dtype}")
                                    # print(f"    Stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
                                    #     f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

                                    if torch.isnan(tensor).any():
                                        print(f"    WARNING: NaN detected in element {i} of layer {layer_name}!")
                                        exit(1)
                                    if torch.isinf(tensor).any():
                                        print(f"    WARNING: Inf detected in element {i} of layer {layer_name}!")
                                        exit(1)

                        elif hasattr(output, 'last_hidden_state'):
                            # Transformer model output (like from DINOv3)
                            # print(f"Layer {layer_name}: (ModelOutput)")
                            if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
                                tensor = output.last_hidden_state
                                # print(f"  last_hidden_state shape: {tensor.shape}, dtype: {tensor.dtype}")
                                # print(f"  Stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
                                #     f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

                                if torch.isnan(tensor).any():
                                    print(f"  WARNING: NaN detected in last_hidden_state of layer {layer_name}!")
                                    exit(1)
                                if torch.isinf(tensor).any():
                                    print(f"  WARNING: Inf detected in last_hidden_state of layer {layer_name}!")
                                    exit(1)
                        else:
                            print(f"Layer {layer_name}: Unknown output type {type(output)}")

                    module.register_forward_hook(hook_fn)

    def _is_dinov3_model(self, model_name: str) -> bool:
        """Check if the model is a DINOv3 model."""
        return 'dinov3' in model_name.lower()

    def _init_backbone(self, model_name: str, global_pool: str, pretrained: bool, frozen: bool, use_group_norm: bool):
        if self._is_dinov3_model(model_name):
            # Handle DINOv3 models using transformers
            if pretrained:
                # Map model names to HuggingFace model IDs
                model_mapping = {
                    'dinov3-vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
                    'dinov3-vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                    'dinov3-vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                }
                hf_model_name = None
                if Path(os.path.expanduser(model_name)).exists():
                    hf_model_name = os.path.expanduser(model_name)
                else:
                    for key, value in model_mapping.items():
                        if key in model_name.lower():
                            hf_model_name = value
                            break

                if hf_model_name is None:
                    # Fallback: assume the model_name is already a HuggingFace model ID
                    hf_model_name = model_name

                logger.info(f"Loading DINOv3 model: {hf_model_name}")
                model = AutoModel.from_pretrained(hf_model_name)
                if hasattr(model.embeddings, "mask_token"):
                    logger.info("Removing unused mask_token from DINOv3 backbone")
                    del model.embeddings.mask_token
                model_normalization_transform = torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            else:
                raise ValueError("DINOv3 models require pretrained=True")
        else:
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

        model.eval()
        model.zero_grad()

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        else:
            model.train()

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

        if self._is_dinov3_model(self.model_name):
            # DINOv3 returns a HuggingFace ModelOutput, not a plain tensor.
            if self.feature_aggregation == 'cls' and self.head_wrist_attention != 'patch':
                return feature.last_hidden_state[:, [0], :]
            elif self.feature_aggregation == 'patch' or self.head_wrist_attention == 'patch':
                num_register_tokens = getattr(self, "dinov3_num_register_tokens", 0)
                return feature.last_hidden_state[:, 1 + num_register_tokens:, :]
            # or use all tokens
            assert self.feature_aggregation is None
            return feature.last_hidden_state

        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls' and self.head_wrist_attention != 'patch':
                return feature[:, [0], :]
            elif self.feature_aggregation == 'patch' or self.head_wrist_attention == 'patch':
                return feature[:, 1:, :]

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

    def _process_dinov3_features(self, model_output):
        """Process DINOv3 model output to extract patch features in the expected format."""
        last_hidden_states = model_output.last_hidden_state  # [B, 1 + num_register + num_patches, hidden_size]

        # Get model config
        pc_model = self.key_model_map[self.pointmap_keys[0]]
        num_register_tokens = pc_model.config.num_register_tokens

        # Extract patch features (skip CLS token and register tokens)
        patch_features_flat = last_hidden_states[:, 1 + num_register_tokens:, :]  # [B, num_patches, hidden_size]

        return patch_features_flat

    def forward(self, obs_dict):
        embeddings = list()
        batched_obs_size = None  # batch_size * n_obs_steps
        wrist_embs = list()
        head_emb = None
        self._last_vit_viz = {}
        self._last_3d_viz = {}
        imgs_copy = {}

        if self.vis_attention and not self.attention_hooks_on and self.head_wrist_attention != 'patch' and len(self.rgb_keys) > 0:
            self.setup_vit_attention_hooks()

        if not self.vis_attention and self.attention_hooks_on:
            self.remove_vit_attention_hooks()

        # process rgb input
        if len(self.rgb_keys) > 0:
            if self.share_rgb_model:
                imgs = list()
                pos_embs = list()
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    if batched_obs_size is None:
                        # Equivalent to (B*To)
                        # Where (B) is batch size
                        # And (To) is n_obs_steps
                        batched_obs_size = img.shape[0]
                    else:
                        assert batched_obs_size == img.shape[0]
                    if self.training:
                        img = self.key_train_transform_map[key](img)
                    else:
                        img = self.key_eval_transform_map[key](img)
                    if self.vis_attention:
                        imgs_copy[key] = img.clone().detach()

                    if self.use_vision_norm:
                        img = self.model_normalization_transform(img)

                    if self.wrist_pos_emb:
                        side = key.split('_')[1]
                        # gripper_pos_key = f'gripper_{side}_eef_pos_wrt_head'
                        gripper_pos_key = f'gripper_{side}_eef_pos'
                        assert gripper_pos_key in obs_dict, f"Expected {gripper_pos_key} in obs_dict for wrist positional embedding"
                        gripper_pos_data = obs_dict[gripper_pos_key].reshape(batched_obs_size, 1, 1, -1)    # (B*T, 1, 1, 3)
                        gripper_pos_emb = self.xyz_proj(gripper_pos_data).squeeze(1)   # (B*T, 1, n_emb)
                        pos_embs.append(gripper_pos_emb)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)  # (N*B*T, C, H, W)
                raw_feature = self.key_model_map['rgb'](imgs)
                feature = self.aggregate_feature(raw_feature)   # (N*B*T, 1, D) or (N*B*T, p, D)
                feature = self.key_projection_map['rgb'](feature)  # (N*B*T, 1, n_emb) or (N*B*T, p, n_emb)
                if self.wrist_pos_emb and len(pos_embs) > 0:
                    pos_embs = torch.cat(pos_embs, dim=0)   # (N*B*T, 1, n_emb)
                    if self.add_pos_emb:
                        feature = feature + pos_embs
                    else:
                        feature = torch.cat([pos_embs, feature], dim=-1)
                emb = feature.reshape(-1, batched_obs_size, *feature.shape[1:])  # (N, B*T, 1, n_emb) or (N, B*T, p, n_emb)
                emb = torch.moveaxis(emb, 0, 1)  # (B*T, N, 1, n_emb) or (B*T, N, p, n_emb)
                # during training, randomly mask out some wrist images
                if self.training and self.wrist_mask_prob > 0.0:
                    keep_prob = 1.0 - self.wrist_mask_prob
                    mask = (torch.rand(emb.shape[0], emb.shape[1], device=emb.device) < keep_prob).float().unsqueeze(-1).unsqueeze(-1)  # (B*T, N, 1, 1)
                    emb = emb * mask
                    if self.vis_attention:
                        imgs_copy[key] *= mask[:, 0, 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # mask the visualization images
                if self.head_wrist_attention != 'none':
                    wrist_embs = emb.reshape(batched_obs_size, -1, emb.shape[-1])   # (B*T, N*1, n_emb) or (B*T, N*p, n_emb)
                else:
                    emb = emb.reshape(batched_obs_size, -1)  # (B*T, N*n_emb) or (B*T, N*p*n_emb)
                    embeddings.append(emb)
            else:
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    if batched_obs_size is None:
                        batched_obs_size = img.shape[0]
                    else:
                        assert batched_obs_size == img.shape[0]
                    if self.training:
                        img = self.key_train_transform_map[key](img)
                    else:
                        img = self.key_eval_transform_map[key](img)
                    if self.vis_attention:
                        imgs_copy[key] = img.clone().detach()

                    if self.use_vision_norm:
                        img = self.model_normalization_transform(img) # apply image normalization

                    raw_feature = self.key_model_map[key](img)
                    feature = self.aggregate_feature(raw_feature)   # (B*T, 1, D) or (B*T, p, D)
                    feature = self.key_projection_map[key](feature)  # (B*T, 1, n_emb) or (B*T, p, n_emb)

                    if self.wrist_pos_emb:
                        side = key.split('_')[1]
                        # gripper_pos_key = f'gripper_{side}_eef_pos_wrt_head'
                        gripper_pos_key = f'gripper_{side}_eef_pos'
                        assert gripper_pos_key in obs_dict, f"Expected {gripper_pos_key} in obs_dict for wrist positional embedding"
                        gripper_pos_data = obs_dict[gripper_pos_key].reshape(batched_obs_size, 1, 1, -1)    # (B*T, 1, 1, 3)
                        gripper_pos_emb = self.xyz_proj(gripper_pos_data).squeeze(1)
                        if self.add_pos_emb:   
                            feature = feature + gripper_pos_emb
                        else:
                            feature = torch.cat([gripper_pos_emb, feature], dim=-1)

                    # during training, randomly mask out some wrist images
                    if self.training and self.wrist_mask_prob > 0.0:
                        keep_prob = 1.0 - self.wrist_mask_prob
                        mask = (torch.rand(feature.shape[0], device=feature.device) < keep_prob).float().unsqueeze(-1).unsqueeze(-1)
                        feature = feature * mask
                        if self.vis_attention:
                            imgs_copy[key] *= mask[:, 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # mask the visualization images
                    if self.head_wrist_attention != 'none':
                        wrist_embs.append(feature)  # list of (B*T, 1, n_emb) or (B*T, p, n_emb)
                    else:
                        feature = feature.reshape(batched_obs_size, -1)  # (B*T, D) or (B*T, p*n_emb)
                        embeddings.append(feature)
                if self.head_wrist_attention != 'none':
                    wrist_embs = torch.concat(wrist_embs, dim=1)  # (B*T, N, n_emb) or (B*T, N*p, n_emb)

        if self.vis_attention and self.attention_hooks_on and self.head_wrist_attention != 'patch' and len(self.rgb_keys) > 0:
            attention_weights = self.get_latest_attention_weights()
            for key, attn in attention_weights.items():
                if 'rgb' in key:  # Only visualize RGB attention
                    rgb_key = key.split('_')[0]  # Extract the RGB key
                    if rgb_key in imgs_copy:
                        rgb_data = imgs_copy[rgb_key]
                    else:
                        rgb_data = torch.cat([img for img in imgs_copy.values()], dim=0)
                    self._last_vit_viz[rgb_key] = self.visualize_vit_attention(attn, rgb_data)
            if not attention_weights and getattr(self, "force_vit_attention", False):
                for rgb_key, rgb_data in imgs_copy.items():
                    if rgb_key in self.key_model_map:
                        model = self.key_model_map[rgb_key]
                    elif "rgb" in self.key_model_map:
                        model = self.key_model_map["rgb"]
                    else:
                        model = None
                    if model is None or not hasattr(model, "get_last_selfattention"):
                        if model is None or not hasattr(model, "config"):
                            logger.warning(f"No attention weights for {rgb_key}; model lacks get_last_selfattention.")
                            continue
                    img = rgb_data
                    if self.use_vision_norm:
                        img = self.model_normalization_transform(img)
                    with torch.no_grad():
                        attn = None
                        if hasattr(model, "get_last_selfattention"):
                            attn = model.get_last_selfattention(img)
                        elif hasattr(model, "config"):
                            if self._is_dinov3_model(self.model_name):
                                try:
                                    model.config._attn_implementation = "eager"
                                except Exception as exc:
                                    logger.warning(f"Failed to set DINOv3 attn_implementation: {exc}")
                                if getattr(model.config, "_attn_implementation", None) != "eager":
                                    logger.warning("DINOv3 attention requires attn_implementation='eager'.")
                                    continue
                            try:
                                model.config.output_attentions = True
                                model.config.return_dict = True
                            except ValueError as exc:
                                logger.warning(f"DINOv3 attention disabled: {exc}")
                                continue
                            output = model(img, output_attentions=True)
                            attn = output.attentions[-1] if output.attentions else None
                        if attn is None:
                            logger.warning(f"No attention weights returned for {rgb_key}.")
                            continue
                    self._last_vit_viz[rgb_key] = self.visualize_vit_attention(attn, rgb_data)

        # Process RGB + xyz pointmap inputs
        if self.pointmap_keys:
            rgb_data = []
            pcd_data = []
            mask_data = []

            for pcd_key in self.pointmap_keys:
                rgb_key = pcd_key.replace('pointmap', 'main_rgb')
                rgb = obs_dict[rgb_key]  # B*T, C, H, W
                pcd = obs_dict[pcd_key] # B*T, C, H, W
                mask = obs_dict.get(
                    f'{pcd_key}_mask',
                    torch.ones(pcd.shape[:1] + (1,) + pcd.shape[2:], device=pcd.device)
                )  # B*T, 1, H, W
                if batched_obs_size is None:
                    # Equivalent to (B*To)
                    # Where (B) is batch size
                    # And (To) is n_obs_steps
                    batched_obs_size = rgb.shape[0]
                else:
                    assert batched_obs_size == rgb.shape[0]
                if self.training and rgb_key in self.key_train_transform_map:
                    if self.train_image_transforms is not None:
                        rgb, pcd, mask = self.train_image_transforms.apply_shared(
                            rgb_key, rgb, pcd, mask
                        )
                    else:
                        rgb = self.key_train_transform_map[rgb_key](rgb)
                elif rgb_key in self.key_eval_transform_map:
                    if self.eval_image_transforms is not None:
                        rgb, pcd, mask = self.eval_image_transforms.apply_shared(
                            rgb_key, rgb, pcd, mask
                        )
                    else:
                        rgb = self.key_eval_transform_map[rgb_key](rgb)
                rgb = rgb.unsqueeze(1) # B, T, C, H, W
                pcd = pcd.unsqueeze(1) # B, T, C, H, W
                mask = mask.unsqueeze(1) # B, T, 1, H, W
                assert rgb.shape == pcd.shape
                rgb_data.append(rgb)
                pcd_data.append(pcd)
                mask_data.append(mask)

            rgb = torch.stack(rgb_data, dim=0)  # n_cam, B, T, C, H, W
            pcd = torch.stack(pcd_data, dim=0)  # n_cam, B, T, C, H, W
            mask = torch.stack(mask_data, dim=0)  # n_cam, B, T, 1, H, W

            # save pointcloud here to debug
            # print(f"pcd shape: {pcd.shape}, rgb shape: {rgb.shape}")
            # save_attention_data(pcd[0, 0, 0, :3, ...].reshape(3, -1).T.cpu().numpy(), np.ones_like(pcd[0, 0, 0, :3, ...].reshape(3, -1).T.cpu().numpy()), './runs/attention_visualization', 'input_debug')

            n_cam, B, T = rgb.shape[:3]
            assert batched_obs_size == B * T, f"batched_obs_size {batched_obs_size} does not match B*T {B*T}"

            rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
            pcd = einops.rearrange(pcd, "ncam b fs c h w -> (b fs ncam) c h w")
            mask = einops.rearrange(mask, "ncam b fs c h w -> (b fs ncam) c h w")
            mask = mask.to(pcd.device)  # ensure mask and point cloud are on the same device

            # Process through backbone
            if self._is_dinov3_model(self.pointcloud_model_name):
                rgb_normalized = self.pc_model_normalization_transform(rgb)    # B*T*n_cam, C, H, W
                outputs = self.key_model_map[pcd_key](pixel_values=rgb_normalized).last_hidden_state  # [B, 1 + num_register + num_patches, hidden_size]
                # Extract patch features (skip CLS token and register tokens)
                rgb_features = outputs[:, 1 + self.key_model_map[pcd_key].config.num_register_tokens:, :]  # [B, num_patches, hidden_size]
            else:
                rgb_normalized = self.pc_model_normalization_transform(rgb)    # B*T*n_cam, C, H, W
                rgb_features = self.key_model_map[pcd_key](rgb_normalized)      # B*T*n_cam, h*w+1, feature_dim
                rgb_features = rgb_features[:, 1:, :]  # Remove the first token (CLS token)
            # rgb_features.shape[1] should be a square number h*w, so we can rearrange it
            assert int(np.sqrt(rgb_features.shape[1]))**2 == rgb_features.shape[1], \
                f"Expected rgb_features shape[1] to be a perfect square, got {rgb_features.shape[1]}"
            
            # check if rgb_features has nan or inf
            # if torch.isnan(rgb_features).any() or torch.isinf(rgb_features).any():
            #     print("NaN or Inf in raw pointmap RGB features!")
            #     print(rgb_features)
            #     # plot and show the imgs and exit
            #     for key in self.rgb_keys:
            #         for i in range(imgs_copy[key].shape[0]):
            #             cv2.imshow('./runs/rgb/input_image_'+key+'_'+str(i), imgs_copy[key][i].permute(1, 2, 0).cpu().numpy()[..., ::-1])    # RGB to BGR
            #             cv2.imwrite('./runs/rgb/input_image_'+key+'_'+str(i)+'.png', imgs_copy[key][i].permute(1, 2, 0).cpu().numpy()[..., ::-1]*255)
            #     exit(1)

            feat_h, feat_w = int(np.sqrt(rgb_features.shape[1])), int(np.sqrt(rgb_features.shape[1]))
            pcd = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")  # B*T*n_cam, C, feat_h, feat_w
            mask = F.interpolate(mask, (feat_h, feat_w), mode="nearest")  # B*T*n_cam, 1, feat_h, feat_w
            pcd = einops.rearrange(
                pcd, "(b fs ncam) c h w -> b fs (ncam h w) c", b=B, ncam=n_cam, fs=T, h=feat_h, w=feat_w
            )   # B, T, n_cam*feat_h*feat_w, C
            mask = einops.rearrange(
                mask, "(b fs ncam) c h w -> b fs (ncam h w) c", b=B, ncam=n_cam, fs=T, h=feat_h, w=feat_w
            )   # B, T, n_cam*feat_h*feat_w, 1
            rgb_features = einops.rearrange(
                rgb_features, "(b fs ncam) (h w) c -> b fs (ncam h w) c", b=B, ncam=n_cam, fs=T, h=feat_h, w=feat_w
            )   # B, T, n_cam*feat_h*feat_w, feature_dim
            # save_attention_data(pcd[0, :3, ...].reshape(-1,3).cpu().numpy(), np.ones_like(pcd[0, :3, ...].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_before_mask_debug')

            # apply pointcloud augmentation
            if self.pointcloud_augmentation is not None:
                pcd, mask = self.pointcloud_augmentation(
                    pcd, mask=mask
                )
            # print(f"mask shape: {mask.shape}, masked points: {torch.sum(~mask)} out of {mask.numel()}")
            # print(f"pcd statistics: {pcd.min()}, {pcd.max()}")
            # mask_hand, pcd_in_left, pcd_in_right = self._crop_point_cloud_with_hands(
            #     pcd,
            #     obs_dict['gripper_head_eef_pos_wrt_left'].unsqueeze(1),
            #     obs_dict['gripper_head_eef_rot_axis_angle_wrt_left'].unsqueeze(1),
            #     obs_dict['gripper_head_eef_pos_wrt_right'].unsqueeze(1),
            #     obs_dict['gripper_head_eef_rot_axis_angle_wrt_right'].unsqueeze(1)
            #     )
            # # print(f"pcd statistics, pcd: {pcd.min()}, {pcd.max()}, pcd_in_left: {pcd_in_left[..., -1].min()}, {pcd_in_left[..., -1].max()}, pcd_in_right: {pcd_in_right[..., -1].min()}, {pcd_in_right[..., -1].max()}")
            # if self.do_crop:
            #     mask = torch.logical_and(mask, mask_hand)
            pcd = pcd * mask  # Apply mask to point cloud     (B, T, n_points, 3)
            rgb_features = rgb_features * mask  # Apply mask to RGB features

            # save_attention_data(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy(), np.ones_like(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_before_downsample_debug')
            # save_attention_data(pcd_in_left[0,0,..., :3].reshape(-1,3).cpu().numpy(), np.ones_like(pcd_in_left[0,0,..., :3].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_in_left_before_downsample_debug')
            # Downsample point cloud
            mask = mask.squeeze(-1)  # B, T, n_points
            pcd, rgb_features, _, mask, _, _ = self._downsample_point_cloud(
                pcd=pcd, rgb_features=rgb_features, mask=mask, num_points=self.num_points,
            )
            # print(f"masked points after downsample: {torch.sum(~mask)} out of {mask.numel()}")
            # pcd shape: B, T, n_points, C; rgb_features shape: B, T, n_points, hidden_dim; mask shape: B, T, n_points
            # save_attention_data(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy(), np.ones_like(pcd[0,0,..., :3].reshape(-1,3).cpu().numpy()), './runs/attention_visualization', 'pcd_after_downsample_debug')

            # Apply position encoding
            pcd_pos_emb = self.xyz_proj(pcd)    # B, T, n_points, hidden_dim
            if self.add_pos_emb:
                assert pcd_pos_emb.shape == rgb_features.shape
                cat_cloud = pcd_pos_emb + rgb_features
            else:
                cat_cloud = torch.cat([pcd_pos_emb, rgb_features], dim=-1)
            # print(f"cat_cloud statistics: min: {cat_cloud.min()}, max: {cat_cloud.max()}, mean: {cat_cloud.mean()}, std: {cat_cloud.std()}, is there nan: {torch.isnan(cat_cloud).any()}, is there inf: {torch.isinf(cat_cloud).any()}")
            if self.vis_attention:
                self._last_3d_viz['feature_map'] = self.visualize_feature_map(
                    cat_cloud, pcd, pcd_data, rgb_data, mask
                )
            if self.vis_attention and self.head_wrist_attention != 'patch':
                pc_emb, attn_weights = self.pc_projection_map(cat_cloud, mask=mask, return_attention=True)  # B, T, n_emb
                self._last_3d_viz['attn_pool'] = self.visualize_attention(
                    attn_weights, pcd, pcd_data, rgb_data, tag_prefix="attn_pool"
                )
            elif self.head_wrist_attention != 'patch':
                pc_emb = self.pc_projection_map(cat_cloud, mask=mask)  # B, T, n_emb
            else:
                pc_emb = self.pc_projection_map(cat_cloud)  # B, T, n_points, n_emb
            pc_emb = pc_emb.reshape(batched_obs_size, *pc_emb.shape[2:])  # (B*T, n_emb) or (B*T, n_points, n_emb)
            if self.debug_nan_checks and (torch.isnan(pc_emb).any() or torch.isinf(pc_emb).any()):
                print("NaN or Inf in pointcloud embedding!")
                # check if any of the rgb or pcd input has nan or inf
                for key in self.pointmap_keys:
                    img = obs_dict[key]
                    if torch.isnan(img).any() or torch.isinf(img).any():
                        print(f"NaN or Inf in input pointmap for key {key}!")
                    rgb_key = key.replace('pointmap', 'main_rgb')
                    img = obs_dict[rgb_key]
                    if torch.isnan(img).any() or torch.isinf(img).any():
                        print(f"NaN or Inf in input rgb for key {rgb_key}!")
            if self.head_wrist_attention != 'none':
                head_emb = pc_emb
            else:
                embeddings.append(pc_emb)

        # process head-wrist attention
        if self.head_wrist_attention != 'none' and self.head_wrist_attention_module is not None:
            assert head_emb is not None and len(wrist_embs) > 0
            if self.vis_attention and self.head_wrist_attention == 'patch':
                attended_features, wrist_attn_weights, head_attn_weights = self.head_wrist_attention_module(
                    wrist_embs,     # (B*T, N*1, n_emb) or (B*T, N*p, n_emb)
                    head_emb,   # (B*T, n_emb) or (B*T, num_points, n_emb)
                    head_mask=mask.reshape(batched_obs_size, -1) if mask is not None else None,     # (B*T, num_points)
                    visualize_attention=True
                )
                self._last_3d_viz['head_attn_pool'] = self.visualize_attention(
                    head_attn_weights.unsqueeze(1), pcd, pcd_data, rgb_data, tag_prefix="head_attn_pool"
                )
                for i, key in enumerate(self.rgb_keys):
                    self._last_vit_viz[key] = self.visualize_vit_attention(wrist_attn_weights[:, i*self.num_patches:(i+1)*self.num_patches], imgs_copy[key])
            elif self.head_wrist_attention == 'patch':
                attended_features = self.head_wrist_attention_module(
                    wrist_embs,     # (B*T, N*1, n_emb) or (B*T, N*p, n_emb)
                    head_emb,   # (B*T, n_emb) or (B*T, num_points, n_emb)
                    head_mask=mask.reshape(batched_obs_size, -1) if mask is not None else None,    # (B*T, num_points)
                )
            else:
                attended_features = self.head_wrist_attention_module(
                    wrist_embs,     # (B*T, N*1, n_emb) or (B*T, N*p, n_emb)
                    head_emb,   # (B*T, n_emb) or (B*T, num_points, n_emb)
                )
            attended_features = attended_features.reshape(batched_obs_size, -1)  # (B*T, N*D)
            if self.debug_nan_checks and (torch.isnan(attended_features).any() or torch.isinf(attended_features).any()):
                print("NaN or Inf in head wrist attention output!")
            embeddings.append(attended_features)

        # process lowdim input
        for key in self.low_dim_keys:
            # (B*T, D_low_dim)
            data = obs_dict[key]
            if batched_obs_size is None:
                batched_obs_size = data.shape[0]
            else:
                assert batched_obs_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key], f"{key} {data.shape} {self.key_shape_map[key]} {data.shape}"
            if self.debug_nan_checks and (torch.isnan(data).any() or torch.isinf(data).any()):
                print(f"NaN or Inf in lowdim input for key {key}!")
            embeddings.append(data)

        result = torch.cat(embeddings, dim=-1)
        # check if result has nan or inf
        if self.debug_nan_checks and (torch.isnan(result).any() or torch.isinf(result).any()):
            print("NaN or Inf in observation encoder output!")
        result_dict = {"features": result}
        return result_dict

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1,) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output["features"].shape) == 2
        output_shape_dict = {key: value.shape[1:] for key, value in example_output.items()}
        return output_shape_dict
    
    def visualize_vit_attention(self, attn_map, rgb):
        """
        Visualize ViT attention maps overlaid on input RGB images.
        
        Args:
            attn_map: Attention weights from ViT model, shape [B, num_heads, num_patches+1, num_patches+1]
                    or [B, num_patches+1, num_patches+1] if already averaged across heads or [B, num_patches] if already processed.
            rgb: Input RGB images, shape [B, C, H, W]
        """
        if isinstance(attn_map, torch.Tensor):
            attn_map = attn_map.detach().cpu().numpy()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        
        # Handle different attention map shapes
        if attn_map.ndim == 2:
            # Already processed: [B, num_patches]
            cls_attention = attn_map[0]  # Shape: [num_patches]
        elif attn_map.ndim == 4:
            # Average across attention heads: [B, H, Q, K] -> [B, Q, K]
            attn_map = attn_map.mean(axis=1)
            # only visualize the first image in the batch for simplicity (assume visualization is done during inference)
            # Get attention from CLS token to all patches (first row, excluding CLS->CLS)
            cls_attention = attn_map[0, 0, 1:]  # Shape: [num_patches]
        elif attn_map.ndim == 3:
            # Already averaged: [B, Q, K]
            cls_attention = attn_map[0, 1:]  # Shape: [num_patches]
        else:
            raise ValueError(f"Unexpected attention map shape: {attn_map.shape}")
        
        B = rgb.shape[0]
        
        # Create output directory
        output_dir = getattr(self, "vit_output_dir", None) or getattr(
            self, "attn_output_dir", "./runs/vit_attention_visualization"
        )
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate patch grid dimensions
        num_patches = cls_attention.shape[0]
        if self._is_dinov3_model(self.model_name) and self.dinov3_num_register_tokens > 0:
            if num_patches > self.dinov3_num_register_tokens:
                cls_attention = cls_attention[self.dinov3_num_register_tokens:]
                num_patches = cls_attention.shape[0]
        patch_size = int(np.sqrt(num_patches))
        if patch_size * patch_size != num_patches:
            trimmed = patch_size * patch_size
            if trimmed == 0:
                raise ValueError(f"Expected square number of patches, got {num_patches}")
            logger.warning(
                f"Non-square patch count {num_patches}; trimming to {trimmed} for visualization."
            )
            cls_attention = cls_attention[:trimmed]
            num_patches = cls_attention.shape[0]
            patch_size = int(np.sqrt(num_patches))
        
        # Reshape attention to spatial grid
        attention_grid = cls_attention.reshape(patch_size, patch_size)
        
        # Normalize attention weights to [0, 1]
        attention_normalized = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
        
        # Prepare input image for visualization
        img = rgb[0]  # Shape: [C, H, W]
        
        # Convert to HWC format and scale to [0, 255]
        img_viz = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Resize attention map to match image size
        img_h, img_w = img_viz.shape[:2]
        attention_resized = cv2.resize(attention_normalized, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        # Create heatmap visualization
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Overlay attention on image
        overlay_alpha = 0.4
        overlay = (overlay_alpha * heatmap + (1 - overlay_alpha) * img_viz).astype(np.uint8)
        
        # Create side-by-side visualization
        vis_images = [
            img_viz,  # Original image
            heatmap,  # Attention heatmap
            overlay   # Overlay
        ]
        
        # Add labels to each image
        labeled_images = []
        labels = ['Original', 'Attention', 'Overlay']
        
        for img, label in zip(vis_images, labels):
            label_height = 30
            label_bar = np.full((label_height, img.shape[1], 3), fill_value=0, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = label_height - 8
            cv2.putText(label_bar, label, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            labeled_img = np.vstack([label_bar, img])
            labeled_images.append(labeled_img)

        combined_viz = np.hstack(labeled_images)
        save_path = os.path.join(output_dir, f'vit_attention_{timestamp}.png')
        cv2.imwrite(save_path, cv2.cvtColor(combined_viz, cv2.COLOR_RGB2BGR))
        print(f"Saved ViT attention visualization to {save_path}")
        return combined_viz

    # Additional helper method to add to your DiTObs3DEncoder class
    def setup_vit_attention_hooks(self):
        """
        Set up hooks to capture attention weights from ViT models.
        Call this method after model initialization if you want to visualize attention.
        """
        self.attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                # input is a tuple; we need the first element
                input = input[0]  # Shape: [B, N, D]
                # Compute QKV using the same method as the attention module
                qkv = module.qkv(input)  # Shape: [B, N, 3*D]
                B, N, D3 = qkv.shape
                assert D3 % 3 == 0, "qkv output dimension is not divisible by 3"
                D = D3 // 3
                assert D % module.num_heads == 0, "qkv dimension is not divisible by num_heads"
                qkv = qkv.reshape(B, N, 3, module.num_heads, D // module.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: [B, num_heads, N, head_dim]
                
                # Compute attention weights: softmax(QK^T / sqrt(head_dim))
                scale = (D // module.num_heads) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
                attn = torch.softmax(attn, dim=-1)
                
                # Store attention weights
                self.attention_weights[name] = attn.detach().cpu()
            return hook
        
        def register_attention_hook(module, name):
            if module is None or not hasattr(module, 'qkv'):
                logger.warning(f"Unable to register attention hook for {name}: module missing qkv.")
                return False
            module.register_forward_hook(attention_hook(name))
            print(f"Registered attention hook for {name}")
            return True

        # Register hooks on attention layers
        models_to_hook = {}
        if self.share_rgb_model:
            models_to_hook['rgb'] = self.key_model_map['rgb']
        else:
            for key in self.rgb_keys:
                if key in self.key_model_map:
                    models_to_hook[key] = self.key_model_map[key]
        for key, model in models_to_hook.items():
            # only hook the last attention layer
            if hasattr(model, 'blocks') and (isinstance(model.blocks, nn.ModuleList) or isinstance(model.blocks, nn.Sequential)):
                last_block = model.blocks[-1]
                if hasattr(last_block, 'attn') and hasattr(last_block.attn, 'qkv'):
                    register_attention_hook(last_block.attn, f"{key}_attn")
            elif hasattr(model, 'encoder'):
                encoder = model.encoder
                layers = None
                if hasattr(encoder, 'layers'):
                    layers = encoder.layers
                elif hasattr(encoder, 'layer'):
                    layers = encoder.layer
                if layers is not None and len(layers) > 0:
                    last_layer = layers[-1]
                    attn_module = None
                    if hasattr(last_layer, 'attention'):
                        attn_module = last_layer.attention
                        if hasattr(attn_module, 'attention'):
                            attn_module = attn_module.attention
                    if attn_module is None:
                        logger.warning(f"Could not locate attention module for {key} encoder layer.")
                        continue
                    if not register_attention_hook(attn_module, f"{key}_attn"):
                        continue
                else:
                    logger.warning(f"Encoder for {key} has no layers attribute; skipping hook registration.")
        self.attention_hooks_on = True
        print("Attention hooks setup complete.")

    def remove_vit_attention_hooks(self):
        """Remove all registered attention hooks."""
        for name, module in self.named_modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
        self.attention_weights = {}
        self.attention_hooks_on = False
        print("Removed all attention hooks.")

    def get_latest_attention_weights(self):
        """Get the most recently captured attention weights."""
        return self.attention_weights

    def set_normalizer(self, normalizer):
        """Store the dataset normalizer for later visualization use."""
        self.obs_normalizer = normalizer
        self._pointmap_denorm_cache.clear()

    def enable_attention_visualization(self):
        self.attn_force_eager = True

    def _get_pointmap_denorm_params(self, key):
        """Extract per-channel scale/offset for the given pointmap key."""
        if key is None or self.obs_normalizer is None:
            return None
        if key in self._pointmap_denorm_cache:
            return self._pointmap_denorm_cache[key]
        try:
            pointmap_normalizer = self.obs_normalizer[key]
        except KeyError:
            logger.warning(f"No normalizer entry found for pointmap key {key}")
            return None
        shape = self.key_shape_map.get(key)
        if shape is None:
            return None
        C, H, W = shape
        scale = pointmap_normalizer.params_dict['scale'].detach().cpu().view(C, H, W)
        offset = pointmap_normalizer.params_dict['offset'].detach().cpu().view(C, H, W)
        scale_channels = scale.mean(dim=(-1, -2))
        offset_channels = offset.mean(dim=(-1, -2))
        max_scale_deviation = (scale - scale_channels[:, None, None]).abs().max().item()
        if max_scale_deviation > 1e-4:
            logger.warning(
                f"Pointmap normalizer for {key} varies spatially (max diff {max_scale_deviation:.2e}); "
                "using channel-wise averages for visualization."
            )
        params = {
            'scale': scale_channels.numpy(),
            'offset': offset_channels.numpy()
        }
        self._pointmap_denorm_cache[key] = params
        return params

    def _unnormalize_points(self, points, key):
        """Unnormalize flattened XYZ points using cached stats."""
        params = self._get_pointmap_denorm_params(key)
        if params is None:
            return points
        scale = params['scale']
        offset = params['offset']
        if isinstance(points, torch.Tensor):
            scale_tensor = torch.as_tensor(scale, dtype=points.dtype, device=points.device)
            offset_tensor = torch.as_tensor(offset, dtype=points.dtype, device=points.device)
            return (points - offset_tensor) / (scale_tensor + 1e-12)
        elif isinstance(points, np.ndarray):
            return (points - offset.astype(points.dtype)) / (scale.astype(points.dtype) + 1e-12)
        else:
            return points

    def visualize_attention(self, attn_weights, pcd, pcd_data, rgb_data, tag_prefix: str = "attention"):
        """ Visualize attention weights for point cloud data. Highlight the points with attention weights.

        Args:
            attn_weights: Attention weights from the model, shape [B, T, n_points]
            pcd: Point cloud data, shape [B, T, n_points, C]
            pcd_data: full point cloud before downsampling, list of n_cam, [B, T, C, H, W]
            rgb_data: RGB data corresponding to the point cloud, list of n_cam, [B, T, C, H, W]
        """
        from hommi.train_network.model.vision_3d.point_cloud_utils import show_point_cloud, show_point_cloud_plt, show_point_cloud_viser

        # Ensure tensors are moved to CPU and converted to numpy
        attn_weights = attn_weights.detach().cpu().numpy()
        pcd = pcd.detach().cpu().numpy()

        B, T, n_points, C = pcd.shape
        assert attn_weights.shape == (B, T, n_points), \
            f"Expected attn_weights shape to be (B, T, n_points), got {attn_weights.shape}"

        # Create output directory if it doesn't exist
        output_dir = getattr(self, "attn_output_dir", "./runs/attention_visualization")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = getattr(self, "vis_output_prefix", "")

        pointmap_key = self.pointmap_keys[0] if len(self.pointmap_keys) > 0 else None

        total_frames = B * T
        if total_frames == 0 or self.max_vis_frames <= 0:
            return None

        num_samples = min(self.max_vis_frames, total_frames)
        flat_indices = self._rng.choice(total_frames, size=num_samples, replace=False)
        frame_imgs = []
        max_h = 0
        max_w = 0

        for flat_idx in flat_indices:
            b = flat_idx // T
            t = flat_idx % T
            tag = f"{prefix}{tag_prefix}_{timestamp}_b{b}_t{t}"

            points_3d = pcd[b, t, :, :3]  # Extract XYZ coordinates
            points_3d = self._unnormalize_points(points_3d, pointmap_key)
            weights = attn_weights[b, t]

            # Apply Gaussian blur to smooth attention weights
            weights_blurred = gaussian_blur_pointcloud(
                points_3d,
                weights,
                sigma_spatial=0.05,
                sigma_intensity=0.1
            )

            # Normalize attention weights to [0, 1] for color mapping
            weights_normalized = (
                (weights_blurred - weights_blurred.min()) /
                (weights_blurred.max() - weights_blurred.min() + 1e-8)
            )

            # Create color map based on attention weights (viridis-like colors)
            colors = create_attention_colormap(weights_normalized)

            save_attention_data(points_3d, colors, output_dir, tag)
            full_pcd = pcd_data[0][b, t, :3, ...].reshape(3, -1).T.cpu().numpy()
            full_pcd = self._unnormalize_points(full_pcd, pointmap_key)
            # save full attention map
            save_attention_data(
                full_pcd,
                rgb_data[0][b, t, :3, ...].reshape(3, -1).T.cpu().numpy(),
                output_dir,
                f"full_{tag}"
            )
            full_colors = interpolate_colors_knn(
                source_points=points_3d,
                source_colors=colors,
                target_points=full_pcd,
                k=5,  # Use 5 nearest neighbors
                max_distance=None
            )
            save_attention_data(full_pcd, full_colors, output_dir, f"full_interpolated_{tag}")

            # Create a matplotlib figure for the point cloud attention
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(121, projection='3d')
            scatter = ax.scatter(
                full_pcd[:, 0],
                full_pcd[:, 1],
                full_pcd[:, 2],
                c=full_colors,
                s=2
            )
            ax.set_title(f'3D Point Cloud Attention (b={b}, t={t})')
            ax.invert_zaxis()
            ax.view_init(elev=45, azim=45)
            # label axes
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            
            # Add a 2D projection view
            ax2 = fig.add_subplot(122, projection='3d')
            scatter2 = ax2.scatter(
                full_pcd[:, 0],
                full_pcd[:, 1],
                full_pcd[:, 2],
                c=full_colors,
                s=2
            )
            ax2.invert_zaxis()
            ax2.view_init(elev=90, azim=-90)  # Top-down view
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title(f'Top-Down Projection (b={b}, t={t})')
            plt.colorbar(scatter2, ax=ax2)

            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
            plt.savefig(os.path.join(output_dir, f'pcd_attention_{tag}.png'))
            plt.close(fig)

            frame_imgs.append(img_array)
            max_h = max(max_h, img_array.shape[0])
            max_w = max(max_w, img_array.shape[1])

        if len(frame_imgs) == 1:
            return frame_imgs[0]

        pad = 8
        padded_imgs = []
        for img in frame_imgs:
            pad_h = max_h - img.shape[0]
            pad_w = max_w - img.shape[1]
            padded = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            padded_imgs.append(padded)
        gap = np.zeros((pad, max_w, 3), dtype=np.uint8)
        combined = padded_imgs[0]
        for img in padded_imgs[1:]:
            combined = np.vstack([combined, gap, img])
        return combined

    def visualize_feature_map(self, features, pcd, pcd_data, rgb_data, mask):
        """Visualize concatenated point features using PCA colors and interpolate to full pointcloud."""
        features = features.detach().cpu().numpy()
        pcd = pcd.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None

        B, T, n_points, _ = pcd.shape
        total_frames = B * T
        if total_frames == 0 or self.max_vis_frames <= 0:
            return None

        output_dir = getattr(self, "attn_output_dir", "./runs/attention_visualization")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = getattr(self, "vis_output_prefix", "")
        pointmap_key = self.pointmap_keys[0] if len(self.pointmap_keys) > 0 else None

        num_samples = min(self.max_vis_frames, total_frames)
        flat_indices = self._rng.choice(total_frames, size=num_samples, replace=False)
        frame_imgs = []
        max_h = 0
        max_w = 0

        for flat_idx in flat_indices:
            b = flat_idx // T
            t = flat_idx % T
            tag = f"{prefix}feature_{timestamp}_b{b}_t{t}"
            points_3d = pcd[b, t, :, :3]
            points_3d = self._unnormalize_points(points_3d, pointmap_key)
            feats = features[b, t]
            if mask_np is not None:
                valid_mask = mask_np[b, t].astype(bool)
                if not np.any(valid_mask):
                    continue
                points_3d = points_3d[valid_mask]
                feats = feats[valid_mask]

            colors = pca_to_rgb(feats, max_points=5000)
            save_attention_data(points_3d, colors, output_dir, tag)

            full_pcd = pcd_data[0][b, t, :3, ...].reshape(3, -1).T.cpu().numpy()
            full_pcd = self._unnormalize_points(full_pcd, pointmap_key)
            full_colors = interpolate_colors_knn(
                source_points=points_3d,
                source_colors=colors,
                target_points=full_pcd,
                k=5,
                max_distance=None
            )
            save_attention_data(full_pcd, full_colors, output_dir, f"full_interpolated_{tag}")

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                full_pcd[:, 0],
                full_pcd[:, 1],
                full_pcd[:, 2],
                c=full_colors,
                s=2
            )
            ax.set_title(f'3D Point Cloud Feature Map (b={b}, t={t})')
            ax.invert_zaxis()
            ax.view_init(elev=45, azim=45)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
            plt.savefig(os.path.join(output_dir, f'pcd_feature_{tag}.png'))
            plt.close(fig)

            frame_imgs.append(img_array)
            max_h = max(max_h, img_array.shape[0])
            max_w = max(max_w, img_array.shape[1])

        if len(frame_imgs) == 1:
            return frame_imgs[0]
        if not frame_imgs:
            return None

        pad = 8
        padded_imgs = []
        for img in frame_imgs:
            pad_h = max_h - img.shape[0]
            pad_w = max_w - img.shape[1]
            padded = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            padded_imgs.append(padded)
        gap = np.zeros((pad, max_w, 3), dtype=np.uint8)
        combined = padded_imgs[0]
        for img in padded_imgs[1:]:
            combined = np.vstack([combined, gap, img])
        return combined

def gaussian_blur_pointcloud(points, weights, sigma_spatial=0.05, sigma_intensity=0.1, k_neighbors=10):
    """
    Apply Gaussian blur to attention weights based on spatial proximity of points.

    Args:
        points: (N, 3) array of point coordinates
        weights: (N,) array of attention weights
        sigma_spatial: spatial standard deviation for Gaussian kernel
        sigma_intensity: intensity standard deviation for bilateral-like filtering
        k_neighbors: number of neighbors to consider for each point

    Returns:
        blurred_weights: (N,) array of smoothed attention weights
    """
    from sklearn.neighbors import NearestNeighbors

    # Build KNN model for efficient neighbor search
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(points)), algorithm='kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    blurred_weights = np.zeros_like(weights)

    for i in range(len(points)):
        neighbor_distances = distances[i][1:]  # Exclude self (distance 0)
        neighbor_indices = indices[i][1:]      # Exclude self

        # Include the point itself
        all_distances = np.concatenate([[0], neighbor_distances])
        all_indices = np.concatenate([[i], neighbor_indices])
        all_weights = weights[all_indices]

        # Spatial Gaussian weights
        spatial_weights = np.exp(-all_distances**2 / (2 * sigma_spatial**2))

        # Optional: Bilateral-like intensity weights (similar intensities get higher weight)
        if sigma_intensity > 0:
            intensity_diff = np.abs(all_weights - weights[i])
            intensity_weights = np.exp(-intensity_diff**2 / (2 * sigma_intensity**2))
            combined_weights = spatial_weights * intensity_weights
        else:
            combined_weights = spatial_weights

        # Normalize weights
        combined_weights = combined_weights / (combined_weights.sum() + 1e-8)

        # Weighted average
        blurred_weights[i] = np.sum(all_weights * combined_weights)

    return blurred_weights

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


def pca_to_rgb(features, max_points=5000):
    """Project features to RGB using PCA on a subset of points."""
    if features.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    feats = features.astype(np.float32, copy=False)
    num_points, feat_dim = feats.shape
    if feat_dim < 3:
        pad = np.zeros((num_points, 3 - feat_dim), dtype=feats.dtype)
        feats = np.concatenate([feats, pad], axis=1)
        feat_dim = feats.shape[1]
    if num_points > max_points:
        idx = np.random.choice(num_points, size=max_points, replace=False)
        sample = feats[idx]
    else:
        sample = feats
    mean = sample.mean(axis=0, keepdims=True)
    sample_centered = sample - mean
    try:
        _, _, vt = np.linalg.svd(sample_centered, full_matrices=False)
        components = vt[:3].T
    except np.linalg.LinAlgError:
        components = np.eye(feat_dim, 3, dtype=feats.dtype)
    projected = (feats - mean) @ components
    min_vals = projected.min(axis=0, keepdims=True)
    max_vals = projected.max(axis=0, keepdims=True)
    denom = np.maximum(max_vals - min_vals, 1e-6)
    normalized = (projected - min_vals) / denom
    return normalized[:, :3].astype(np.float32)


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
