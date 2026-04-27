import torch
import torch.nn as nn
import einops
import numpy as np

from typing import List, Type
import math


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class MaxExtractor(nn.Module):
    """
    Extracts from a point cloud by passing through an MLP followed by a max pooling.
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 block_channel=(64, 128, 256, 512),
                 **kwargs
                 ):
        """
        Args:
            in_shape: (int, int) or (int, int, int)
                - (n_points, in_channels) if 2D
                - (n_points, frame_stack, in_channels) if 3D
            out_channels: int
            use_layernorm: bool
            final_norm: str
        """
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, pc, mask=None):
        # assume pc has dim batch, frame_stack, n, d
        x = self.mlp(pc)
        if mask is not None:
            x.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
        x = torch.max(x, 2)[0]

        x = self.final_projection(x)
        return x
    

class AttentionExtractor(nn.Module):
    """
    Extracts from a point cloud by passing through an MLP followed by a attention pooling.
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 hidden_dim=256,
                 num_heads=4,
                 **kwargs
                 ):
        super().__init__()
        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # Create learnable key for each head
        self.queries = nn.Parameter(torch.randn(num_heads, hidden_dim))
        
        # Query projection MLP
        self.K_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Value projection MLP
        self.V_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(hidden_dim * num_heads, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(m.weight.data, gain)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
        
    def forward(self, pc, mask=None, return_attention=False):
        # assume pc has dim batch, frame_stack, n, d
        B, F, N, D = pc.shape
        
        # Project to query and value
        K = self.K_mlp(pc)  # [B, F, N, hidden_dim]
        V = self.V_mlp(pc)  # [B, F, N, hidden_dim]
        
        # Expand key for broadcasting
        Q = self.queries.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
        
        # Compute attention using scaled_dot_product_attention
        if mask is not None:
            # all_masked = ~mask.any(dim=-1)  # [B, F] - True where all points are masked
            
            # if all_masked.any():
            #     # Create output tensor filled with zeros for fully masked frames
            #     x = torch.zeros(B, F, self.out_channels, device=pc.device, dtype=pc.dtype)
                
            #     # Process only frames that have at least one valid point
            #     valid_frames = ~all_masked
                
            #     if valid_frames.any():
            #         # Get indices of valid frames
            #         valid_batch_idx, valid_frame_idx = torch.where(valid_frames)
                    
            #         # Process valid frames
            #         valid_pc = pc[valid_batch_idx, valid_frame_idx]  # [num_valid, N, D]
            #         valid_mask = mask[valid_batch_idx, valid_frame_idx]  # [num_valid, N]
                    
            #         # Recompute K, V for valid frames only
            #         valid_K = self.K_mlp(valid_pc)  # [num_valid, N, hidden_dim]
            #         valid_V = self.V_mlp(valid_pc)  # [num_valid, N, hidden_dim]
                    
            #         # Expand Q for valid frames
            #         valid_Q = self.queries.unsqueeze(0).expand(len(valid_batch_idx), -1, -1)  # [num_valid, num_heads, hidden_dim]
                    
            #         # Prepare mask for attention
            #         valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, self.num_heads, -1)  # [num_valid, num_heads, N]
                    
            #         # Compute attention using scaled_dot_product_attention
            #         attn_output = torch.nn.functional.scaled_dot_product_attention(
            #             valid_Q.unsqueeze(2),  # [num_valid, num_heads, 1, hidden_dim]
            #             valid_K.unsqueeze(1),  # [num_valid, 1, N, hidden_dim]
            #             valid_V.unsqueeze(1),  # [num_valid, 1, N, hidden_dim]
            #             attn_mask=valid_mask_expanded.unsqueeze(2),  # [num_valid, num_heads, 1, N]
            #             dropout_p=0.0,
            #             is_causal=False
            #         )  # [num_valid, num_heads, 1, hidden_dim]
                    
            #         # Combine heads
            #         valid_x = einops.rearrange(attn_output.squeeze(2), 'b h d -> b (h d)')  # [num_valid, hidden_dim * num_heads]
                    
            #         # Final projection
            #         valid_x = self.final_projection(valid_x)  # [num_valid, out_channels]
                    
            #         # Place results back in the output tensor
            #         x[valid_batch_idx, valid_frame_idx] = valid_x
                
            #     if return_attention:
            #         # Create attention weights tensor
            #         attn_weight = torch.zeros(B, F, N, device=pc.device, dtype=pc.dtype)
                    
            #         if valid_frames.any():
            #             # Compute attention weights for valid frames
            #             valid_batch_idx, valid_frame_idx = torch.where(valid_frames)
            #             valid_K = K[valid_batch_idx, valid_frame_idx]
            #             valid_mask = mask[valid_batch_idx, valid_frame_idx]
                        
            #             scale_factor = 1.0 / math.sqrt(self.head_dim)
            #             valid_Q_expanded = self.queries.unsqueeze(0).expand(len(valid_batch_idx), -1, -1)
                        
            #             # Compute attention scores
            #             scores = torch.einsum('bhd,bnd->bhn', valid_Q_expanded, valid_K) * scale_factor
                        
            #             # Apply mask
            #             scores.masked_fill_(~valid_mask.unsqueeze(1), float('-inf'))
                        
            #             # Apply softmax and average over heads
            #             valid_attn = torch.softmax(scores, dim=-1).mean(dim=1)  # [num_valid, N]
            #             attn_weight[valid_batch_idx, valid_frame_idx] = valid_attn
                    
            #         return x, attn_weight
                
            #     return x
            # else:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(2)
        
        # Use scaled_dot_product_attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, F, N, num_heads, head_dim]
        
        # Combine heads
        x = einops.rearrange(attn_output, 'b f h d -> b f (h d)')  # [B, F, hidden_dim]
        
        # Final projection
        x = self.final_projection(x)  # [B, F, out_channels]

        if return_attention:
            scale_factor = 1.0 / math.sqrt(self.head_dim)
            attn_bias = torch.zeros(B, F, self.num_heads, N, device=pc.device)
            attn_bias.masked_fill_(mask.logical_not(), float('-inf'))
            attn_weight = Q @ K.transpose(-2, -1) * scale_factor  # [B, F, num_heads, N]
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)  # [B, F, num_heads, N]
            attn_weight = attn_weight.mean(dim=-2)  # [B, F, N]
            return x, attn_weight  # [B, F, out_channels], [B, F, N]
        if torch.isnan(attn_output).any():
            print("NAN in attention weights!")
        return x
    

class MultiModalSelfAttention(nn.Module):
    """Self-attention encoder for multimodal feature tokens (left wrist, right wrist, head)."""

    def __init__(self, n_emb: int = 768, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_emb = n_emb
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=num_heads,
            # dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Learnable positional embeddings for each modality
        # self.modality_embeddings = nn.Parameter(torch.randn(3, n_emb))  # left, right, head

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
    
    def forward(self, wrist_feats, head_feat, mask=None):
        """
        Args:
            wrist_feats: [B, 2, n_emb]
            head_feat: [B, n_emb]
            mask: [B, 3] optional mask for modalities
        Returns:
            fused_features: [B, 3, n_emb] - aggregated multimodal features
        """
        # Stack features and add modality embeddings
        features = torch.cat([wrist_feats, head_feat.unsqueeze(1)], dim=1)  # [B, 3, n_emb]
        # features = features + self.modality_embeddings.unsqueeze(0)  # Add positional embeddings
        
        # Apply transformer
        attended_features = self.transformer(features, src_key_padding_mask=mask)  # [B, 3, n_emb]
        return attended_features


class WristHeadCrossAttention(nn.Module):
    """Cross-attention between wrist patches and head patches."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, wrist_num_patches: int = 196):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.wrist_num_patches = wrist_num_patches

        # Cross-attention layers
        self.wrist_to_head_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.head_to_wrist_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.ln_wrist = nn.LayerNorm(hidden_dim)
        self.ln_head = nn.LayerNorm(hidden_dim)

        # add an additional token to represent the global feature
        self.wrist_global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.head_global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.wrist_pos_emb = nn.Parameter(torch.randn(1+2*wrist_num_patches, hidden_dim))  # 392 patches + 1 global token

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
    
    def forward(self, wrist_patches, head_patches, wrist_mask=None, head_mask=None, visualize_attention=False):
        """
        Args:
            wrist_patches: [B, 392, hidden_dim] - concatenated left+right wrist patches
            head_patches: [B, 512, hidden_dim] - head patches
            wrist_mask: [B, 392] - mask for wrist patches
            head_mask: [B, 512] - mask for head patches
        Returns:
            global_features: [B, 2*hidden_dim] - concatenated global features from wrist and head
            (optionally) attn_weights: [B, 392], [B, 512] - attention weights of global tokens for visualization
        """
        B = wrist_patches.shape[0]
        # append global token
        wrist_patches = torch.cat(
            [wrist_patches, self.wrist_global_token.expand(B, -1, -1)], dim=1
        )  # [B, 393, hidden_dim]
        # add positional embedding
        assert wrist_patches.size(1) == self.wrist_pos_emb.size(0), f"wrist_patches size {wrist_patches.size(1)} does not match positional embedding size {self.wrist_pos_emb.size(0)}"
        wrist_patches = wrist_patches + self.wrist_pos_emb.unsqueeze(0)  # [B, 393, hidden_dim]
        head_patches = torch.cat(
            [head_patches, self.head_global_token.expand(B, -1, -1)], dim=1
        )  # [B, 513, hidden_dim]
        if wrist_mask is not None:
            wrist_mask = torch.cat(
                [wrist_mask, torch.zeros(B, 1, device=wrist_mask.device, dtype=wrist_mask.dtype)],
                dim=1
            )  # [B, 393]
        if head_mask is not None:
            head_mask = torch.cat(
                [head_mask, torch.zeros(B, 1, device=head_mask.device, dtype=head_mask.dtype)],
                dim=1
            )  # [B, 513]
        # Cross-attention: wrist queries attend to head
        wrist_attended, wrist_attn_weights = self.wrist_to_head_attn(
            query=wrist_patches,
            key=head_patches,
            value=head_patches,
            key_padding_mask=head_mask,
            need_weights=visualize_attention,
            average_attn_weights=True
        )   # [B, 393, hidden_dim], [B, 393, 513]
        wrist_patches = self.ln_wrist(wrist_patches + wrist_attended)

        # Cross-attention: head queries attend to wrist
        head_attended, head_attn_weights = self.head_to_wrist_attn(
            query=head_patches,
            key=wrist_patches,
            value=wrist_patches,
            key_padding_mask=wrist_mask,
            need_weights=visualize_attention,
            average_attn_weights=True
        )   # [B, 513, hidden_dim], [B, 513, 393]
        head_patches = self.ln_head(head_patches + head_attended)

        # get global features
        wrist_global_feat = wrist_patches[:, -1, :]  # [B, hidden_dim]
        head_global_feat = head_patches[:, -1, :]      # [B, hidden_dim]
        global_features = torch.cat([wrist_global_feat, head_global_feat], dim=-1)  # [B, 2*hidden_dim]
        if visualize_attention:
            head_attn_weights = head_attn_weights[..., :-1, -1]  # [B, 513, 393] -> [B, 512]
            wrist_attn_weights = wrist_attn_weights[..., :-1, -1]  # [B, 393, 513] -> [B, 392]
            return global_features, wrist_attn_weights, head_attn_weights
        return global_features


class MaskedImageTokenizer(nn.Module):
    """Handles masked training for wrist images."""
    
    def __init__(self, mask_ratio: float = 0.75, mask_token_learnable: bool = True, hidden_dim: int = 768):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token_learnable = mask_token_learnable
        
        if mask_token_learnable:
            self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        else:
            self.register_buffer('mask_token', torch.zeros(hidden_dim))
    
    def forward(self, x, force_mask=True):
        """
        Args:
            x: [B, N, D] - patch features
            force_mask: whether to apply masking (set False during inference)
        Returns:
            masked_x: [B, N, D] - features with masked patches replaced
            mask: [B, N] - binary mask (1 = masked, 0 = keep)
        """
        if not force_mask or not self.training:
            return x, torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        
        B, N, D = x.shape
        num_masked = int(self.mask_ratio * N)
        
        # Generate random mask for each sample in batch
        mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        for b in range(B):
            masked_indices = torch.randperm(N, device=x.device)[:num_masked]
            mask[b, masked_indices] = True
        
        # Replace masked patches with mask token
        masked_x = x.clone()
        masked_x[mask] = self.mask_token.to(x.dtype)
        
        return masked_x, mask