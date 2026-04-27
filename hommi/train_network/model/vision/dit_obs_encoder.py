"""
This is a modified version of the transformer_obs_encoder in UMI diffusion_policy.
It adds the following features:
- use vision normalization
- use separate train_image_transforms and eval_image_transforms rather than single augmentation
"""

import copy
from typing import Optional

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from transformers import AutoModel
import cv2

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules
from hommi.train_network.common.augmentation import ImageAugmentation

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class DiTObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            train_image_transforms: Optional[ImageAugmentation]=None,
            eval_image_transforms: Optional[ImageAugmentation]=None,
            pretrained: bool=False,
            frozen: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            use_vision_norm: bool=True,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_train_transform_map = nn.ModuleDict()
        key_eval_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        model, model_normalization_transform = self._init_backbone(
            model_name=model_name,
            global_pool=global_pool,
            pretrained=pretrained,
            frozen=frozen,
            use_group_norm=use_group_norm
        )
        self.model_name = model_name
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit') or self._is_dinov3_model(model_name):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation is None:
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(model, nn.Module)
            key_model_map["rgb"] = model

        for key, attr in obs_shape_meta.items():
            if obs_shape_meta[key].get('ignore_by_policy', False):
                continue
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                if not share_rgb_model:
                    this_model = model if share_rgb_model else copy.deepcopy(model)
                    key_model_map[key] = this_model
                key_train_transform_map[key] = train_image_transforms.get_transform(key) if train_image_transforms is not None else nn.Identity()
                key_eval_transform_map[key] = eval_image_transforms.get_transform(key) if eval_image_transforms is not None else nn.Identity()
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_train_transform_map = key_train_transform_map
        self.key_eval_transform_map = key_eval_transform_map
        self.model_normalization_transform = model_normalization_transform
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.use_vision_norm = use_vision_norm
        self.dinov3_num_register_tokens = (
            model.config.num_register_tokens if self._is_dinov3_model(model_name) else 0
        )
        self.vis_attention = False
        self.attention_hooks_on = False
        self.attention_weights = {}
        self._last_vit_viz = {}
        self.force_vit_attention = False
        self.attn_force_eager = False

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _is_dinov3_model(self, model_name: str) -> bool:
        return 'dinov3' in model_name.lower()

    def _init_backbone(self, model_name: str, global_pool: str, pretrained: bool, frozen: bool, use_group_norm: bool):
        if self._is_dinov3_model(model_name):
            if pretrained:
                model_mapping = {
                    'dinov3-vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
                    'dinov3-vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                    'dinov3-vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                }
                hf_model_name = None
                for key, value in model_mapping.items():
                    if key in model_name.lower():
                        hf_model_name = value
                        break

                if hf_model_name is None:
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
                global_pool=global_pool,
                num_classes=0
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
            if self.feature_aggregation == 'cls':
                return feature.last_hidden_state[:, [0], :]
            if self.feature_aggregation == 'patch':
                return feature.last_hidden_state[:, 1 + self.dinov3_num_register_tokens:, :]
            assert self.feature_aggregation is None
            return feature.last_hidden_state
        
        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]
            
            # or use all tokens
            assert self.feature_aggregation is None 
            return feature
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

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

    def visualize_vit_attention(self, attn_map, rgb):
        if isinstance(attn_map, torch.Tensor):
            attn_map = attn_map.detach().cpu().numpy()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()

        if attn_map.ndim == 2:
            cls_attention = attn_map[0]
        elif attn_map.ndim == 4:
            attn_map = attn_map.mean(axis=1)
            cls_attention = attn_map[0, 0, 1:]
        elif attn_map.ndim == 3:
            cls_attention = attn_map[0, 1:]
        else:
            raise ValueError(f"Unexpected attention map shape: {attn_map.shape}")

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

        attention_grid = cls_attention.reshape(patch_size, patch_size)
        attn_min = attention_grid.min()
        attn_max = attention_grid.max()
        attention_normalized = (attention_grid - attn_min) / (attn_max - attn_min + 1e-8)

        img = rgb[0]
        img_viz = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        img_h, img_w = img_viz.shape[:2]
        attention_resized = cv2.resize(
            attention_normalized, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )
        heatmap = cv2.applyColorMap(
            (attention_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )
        overlay_alpha = 0.4
        overlay = (overlay_alpha * heatmap + (1 - overlay_alpha) * img_viz).astype(np.uint8)

        labels = ["Original", "Attention", "Overlay"]
        vis_images = [img_viz, heatmap, overlay]
        labeled_images = []
        for img, label in zip(vis_images, labels):
            label_height = 30
            label_bar = np.full((label_height, img.shape[1], 3), fill_value=0, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = label_height - 8
            cv2.putText(label_bar, label, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            labeled_images.append(np.vstack([label_bar, img]))
        return np.hstack(labeled_images)

    def enable_attention_visualization(self):
        self.attn_force_eager = True

    def setup_vit_attention_hooks(self):
        self.attention_weights = {}

        def attention_hook(name):
            def hook(module, inputs, _output):
                inputs = inputs[0]
                if hasattr(module, "qkv"):
                    qkv = module.qkv(inputs)
                    bsz, num_tokens, dim3 = qkv.shape
                    dim = dim3 // 3
                    qkv = qkv.reshape(
                        bsz, num_tokens, 3, module.num_heads, dim // module.num_heads
                    ).permute(2, 0, 3, 1, 4)
                    q, k = qkv[0], qkv[1]
                    scale = (dim // module.num_heads) ** -0.5
                elif hasattr(module, "query") and hasattr(module, "key"):
                    q = module.query(inputs)
                    k = module.key(inputs)
                    bsz, num_tokens, dim = q.shape
                    num_heads = getattr(module, "num_attention_heads", None)
                    if num_heads is None:
                        num_heads = getattr(module, "num_heads", None)
                    if num_heads is None:
                        raise ValueError("Attention module missing num_heads/num_attention_heads")
                    head_dim = dim // num_heads
                    q = q.reshape(bsz, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
                    k = k.reshape(bsz, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
                    scale = head_dim ** -0.5
                else:
                    raise ValueError("Attention module missing qkv or query/key projections")
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = torch.softmax(attn, dim=-1)
                self.attention_weights[name] = attn.detach().cpu()
            return hook
        
        def register_attention_hook(module, name):
            if module is None or not (hasattr(module, "qkv") or (hasattr(module, "query") and hasattr(module, "key"))):
                logger.warning(f"Unable to register attention hook for {name}: module missing qkv/query/key.")
                return False
            module.register_forward_hook(attention_hook(name))
            return True

        models_to_hook = {}
        if self.share_rgb_model:
            if "rgb" in self.key_model_map:
                models_to_hook["rgb"] = self.key_model_map["rgb"]
        else:
            for key in self.rgb_keys:
                if key in self.key_model_map:
                    models_to_hook[key] = self.key_model_map[key]

        registered = False
        for key, model in models_to_hook.items():
            if hasattr(model, "blocks") and (
                isinstance(model.blocks, nn.ModuleList) or isinstance(model.blocks, nn.Sequential)
            ):
                last_block = model.blocks[-1]
                if hasattr(last_block, "attn"):
                    registered = register_attention_hook(last_block.attn, f"{key}_attn") or registered
                    continue
            if hasattr(model, "encoder"):
                encoder = model.encoder
                layers = None
                if hasattr(encoder, "layers"):
                    layers = encoder.layers
                elif hasattr(encoder, "layer"):
                    layers = encoder.layer
                if layers is not None and len(layers) > 0:
                    last_layer = layers[-1]
                    attn_module = None
                    if hasattr(last_layer, "attention"):
                        attn_module = last_layer.attention
                        if hasattr(attn_module, "attention"):
                            attn_module = attn_module.attention
                    if attn_module is None:
                        logger.warning(f"Could not locate attention module for {key} encoder layer.")
                        continue
                    registered = register_attention_hook(attn_module, f"{key}_attn") or registered
                    continue
            # Fallback: scan for the last attention module with qkv or query/key.
            fallback_module = None
            fallback_name = None
            for _name, module in model.named_modules():
                if hasattr(module, "qkv") or (hasattr(module, "query") and hasattr(module, "key")):
                    fallback_module = module
                    fallback_name = _name
            if fallback_module is not None:
                registered = register_attention_hook(fallback_module, f"{key}_attn") or registered
                continue
            candidate_names = []
            for _name, module in model.named_modules():
                if hasattr(module, "qkv") or hasattr(module, "query") or hasattr(module, "key"):
                    candidate_names.append(_name)
                if len(candidate_names) >= 5:
                    break
            logger.warning(
                f"Model for {key} has no supported attention blocks; "
                f"sample candidates={candidate_names}"
            )
        self.attention_hooks_on = registered

    def remove_vit_attention_hooks(self):
        for _name, module in self.named_modules():
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
        self.attention_weights = {}
        self.attention_hooks_on = False

    def get_latest_attention_weights(self):
        return self.attention_weights
        
    def forward(self, obs_dict):
        embeddings = list()
        batched_obs_size = None  # batch_size * n_obs_steps
        self._last_vit_viz = {}
        imgs_copy = {}

        if (
            self.vis_attention
            and not self.attention_hooks_on
            and len(self.rgb_keys) > 0
            and not self._is_dinov3_model(self.model_name)
        ):
            self.setup_vit_attention_hooks()
        if not self.vis_attention and self.attention_hooks_on:
            self.remove_vit_attention_hooks()
        
        # process rgb input
        if self.share_rgb_model:
            imgs = list()
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
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)  # (N*B*T, C, H, W)
            if self.vis_attention and self._is_dinov3_model(self.model_name):
                use_attn = True
                if hasattr(self.key_model_map["rgb"], "config"):
                    if self.attn_force_eager:
                        try:
                            self.key_model_map["rgb"].config._attn_implementation = "eager"
                        except Exception as exc:
                            logger.warning(f"Failed to set DINOv3 attn_implementation: {exc}")
                    if getattr(self.key_model_map["rgb"].config, "_attn_implementation", None) != "eager":
                        logger.warning("DINOv3 attention requires attn_implementation='eager'.")
                        use_attn = False
                    if use_attn:
                        try:
                            self.key_model_map["rgb"].config.output_attentions = True
                            self.key_model_map["rgb"].config.return_dict = True
                        except ValueError as exc:
                            logger.warning(f"DINOv3 attention disabled: {exc}")
                            use_attn = False
                if use_attn:
                    raw_feature = self.key_model_map["rgb"](imgs, output_attentions=True)
                    if hasattr(raw_feature, "attentions") and raw_feature.attentions:
                        attn = raw_feature.attentions[-1]
                        attn_chunks = []
                        if attn.shape[0] == batched_obs_size * len(self.rgb_keys):
                            attn_chunks = torch.split(attn, batched_obs_size, dim=0)
                        elif attn.shape[0] == batched_obs_size:
                            attn_chunks = [attn] * len(self.rgb_keys)
                        for rgb_key, attn_chunk in zip(self.rgb_keys, attn_chunks):
                            if rgb_key not in imgs_copy:
                                continue
                            try:
                                self._last_vit_viz[rgb_key] = self.visualize_vit_attention(attn_chunk, imgs_copy[rgb_key])
                            except Exception as exc:
                                logger.warning(f"Failed to render vit attention for {rgb_key}: {exc}")
                    else:
                        logger.warning("DINOv3 model did not return attentions; cannot render.")
                else:
                    raw_feature = self.key_model_map["rgb"](imgs)
            else:
                raw_feature = self.key_model_map['rgb'](imgs)
            feature = self.aggregate_feature(raw_feature)   # (N*B*T, D)
            emb = feature.reshape(-1, batched_obs_size, *feature.shape[1:])  # (N, B*T, D)
            emb = torch.moveaxis(emb, 0, 1)  # (B*T, N, D)
            emb = emb.reshape(batched_obs_size, -1)  # (B*T, N*D)
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

                if self.vis_attention and self._is_dinov3_model(self.model_name):
                    use_attn = True
                    if hasattr(self.key_model_map[key], "config"):
                        if self.attn_force_eager:
                            try:
                                self.key_model_map[key].config._attn_implementation = "eager"
                            except Exception as exc:
                                logger.warning(f"Failed to set DINOv3 attn_implementation: {exc}")
                        if getattr(self.key_model_map[key].config, "_attn_implementation", None) != "eager":
                            logger.warning("DINOv3 attention requires attn_implementation='eager'.")
                            use_attn = False
                        if use_attn:
                            try:
                                self.key_model_map[key].config.output_attentions = True
                                self.key_model_map[key].config.return_dict = True
                            except ValueError as exc:
                                logger.warning(f"DINOv3 attention disabled: {exc}")
                                use_attn = False
                    if use_attn:
                        raw_feature = self.key_model_map[key](img, output_attentions=True)
                        if hasattr(raw_feature, "attentions") and raw_feature.attentions:
                            attn = raw_feature.attentions[-1]
                            try:
                                self._last_vit_viz[key] = self.visualize_vit_attention(attn, imgs_copy[key])
                            except Exception as exc:
                                logger.warning(f"Failed to render vit attention for {key}: {exc}")
                        else:
                            logger.warning("DINOv3 model did not return attentions; cannot render.")
                    else:
                        raw_feature = self.key_model_map[key](img)
                else:
                    raw_feature = self.key_model_map[key](img)
                feature = self.aggregate_feature(raw_feature)   # (B*T, D)
                embeddings.append(feature)

        if self.vis_attention and self.attention_hooks_on and len(self.rgb_keys) > 0:
            attention_weights = self.get_latest_attention_weights() or {}
            for key, attn in attention_weights.items():
                if "rgb" not in key:
                    continue
                rgb_key = key.split("_")[0]
                if rgb_key in imgs_copy:
                    rgb_data = imgs_copy[rgb_key]
                    out_key = rgb_key
                elif imgs_copy:
                    rgb_data = torch.cat(list(imgs_copy.values()), dim=0)
                    out_key = self.rgb_keys[0]
                else:
                    continue
                try:
                    self._last_vit_viz[out_key] = self.visualize_vit_attention(attn, rgb_data)
                except Exception as exc:
                    logger.warning(f"Failed to render vit attention for {out_key}: {exc}")
            if not attention_weights and self.force_vit_attention and not self._is_dinov3_model(self.model_name):
                for rgb_key, rgb_data in imgs_copy.items():
                    if rgb_key in self.key_model_map:
                        model = self.key_model_map[rgb_key]
                    elif "rgb" in self.key_model_map:
                        model = self.key_model_map["rgb"]
                    else:
                        model = None
                    if model is None or not hasattr(model, "get_last_selfattention"):
                        logger.warning(f"No attention weights for {rgb_key}; model lacks get_last_selfattention.")
                        continue
                    img = rgb_data
                    if self.use_vision_norm:
                        img = self.model_normalization_transform(img)
                    attn = model.get_last_selfattention(img)
                    try:
                        self._last_vit_viz[rgb_key] = self.visualize_vit_attention(attn, rgb_data)
                    except Exception as exc:
                        logger.warning(f"Failed to render vit attention for {rgb_key}: {exc}")

        # process lowdim input
        for key in self.low_dim_keys:
            # (B*T, D_low_dim)
            data = obs_dict[key]
            if batched_obs_size is None:
                batched_obs_size = data.shape[0]
            else:
                assert batched_obs_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            embeddings.append(data)

        result = torch.cat(embeddings, dim=-1)
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
