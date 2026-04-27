# Diffusion Transformer architectures based on:
# https://arxiv.org/pdf/2212.09748 for the AdaLN
# https://arxiv.org/pdf/2312.04557 for the cross-attention variation

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers."""

    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Attention(nn.Module):
    """Multiheaded self-attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
        causal_block=1,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.causal_block = causal_block

        if is_causal and causal_block > 1:
            print("Disabling torch spda kernel for block causal attention")
            self.use_sdpa = False
        else:
            self.use_sdpa = use_sdpa

        if not self.use_sdpa:
            self.causal_block_mat = nn.Parameter(
                torch.ones((causal_block, causal_block)).bool(),
                requires_grad=False,
            )

    def forward(self, x, pos_embed=None):
        B, N, D = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.proj_drop.p,
                    is_causal=self.is_causal,
                )
        else:
            attn = q @ (k.transpose(-2, -1) / (self.head_dim**0.5))
            if self.is_causal:
                assert N % self.causal_block == 0
                num_blocks = N // self.causal_block
                block_diag_mat = torch.block_diag(
                    *[self.causal_block_mat for _ in range(num_blocks)]
                )
                triu_mat = torch.triu(
                    torch.ones(N, N, device=x.device), diagonal=1
                ).bool()
                mask = torch.logical_and(~block_diag_mat, triu_mat)
                attn = attn.masked_fill(mask.view(1, 1, N, N), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):
    """Multiheaded self-attention block.

    Combines an attention layer and an MLP with residual connections.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act=nn.GELU,
        norm=nn.LayerNorm,
        is_causal=False,
        causal_block=1,
    ):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            causal_block=causal_block,
        )
        self.norm2 = norm(dim)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )

    def forward(self, x, pos_embed=None):
        x = x + self.attn(self.norm1(x), pos_embed)
        x = x + self.mlp(self.norm2(x))
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiTBuildingBlock(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        activation=lambda: nn.GELU(approximate="tanh"),
        use_rms_norm=False,
        is_causal=False,
        causal_block=1,
    ):
        super().__init__()
        self.embed_dim = dim
        self.cond_dim = cond_dim

        if use_rms_norm:
            self.pre_norm = RMSNorm(dim, eps=1e-6)
            self.post_norm = RMSNorm(dim, eps=1e-6)
        else:
            self.pre_norm = nn.LayerNorm(
                dim, elementwise_affine=False, eps=1e-6
            )
            self.post_norm = nn.LayerNorm(
                dim, elementwise_affine=False, eps=1e-6
            )

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            causal_block=causal_block,
        )
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=activation,
            drop=drop,
        )
        self.ada_ln_chunks = 6
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, self.ada_ln_chunks * dim, bias=True)
        )

    def forward(self, x, condition):
        # Split modulation into ATTN and MLP parameters
        adaln_modulation = self.ada_ln_modulation(condition)
        (attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate) = (
            adaln_modulation.chunk(self.ada_ln_chunks, dim=1)
        )

        # Attention block with gating
        modulated_pre_attn = self.scale_and_shift(
            self.pre_norm(x), attn_shift, attn_scale
        )
        attn_output = self.attn(modulated_pre_attn)
        x = x + attn_gate.unsqueeze(1) * attn_output

        # MLP block with gating
        modulated_post_attn = self.scale_and_shift(
            self.post_norm(x), mlp_shift, mlp_scale
        )
        mlp_output = self.mlp(modulated_post_attn)
        x = x + mlp_gate.unsqueeze(1) * mlp_output

        return x

    def scale_and_shift(self, x, shift, scale):
        result = x * (1 + scale.unsqueeze(1))
        result += shift.unsqueeze(1)
        return result


class FinalLayer(nn.Module):
    """
    The final layer for the action DiT
    """

    def __init__(
        self,
        dim,
        cond_dim,
        use_rms_norm=False,
    ):
        super().__init__()
        self.embed_dim = dim
        self.cond_dim = cond_dim

        if use_rms_norm:
            self.norm = RMSNorm(dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.ada_ln_chunks = 2
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, self.ada_ln_chunks * dim),
        )
        self.final_linear = nn.Linear(dim, dim)

    def forward(self, x, cond):
        shift_adaln, scale_adaln = self.ada_ln_modulation(cond).chunk(
            self.ada_ln_chunks, dim=1
        )
        norm_modulated = self.scale_and_shift(
            self.norm(x), shift_adaln, scale_adaln
        )
        return self.final_linear(norm_modulated)

    def scale_and_shift(self, x, shift, scale):
        result = x * (1 + scale.unsqueeze(1))
        result += shift.unsqueeze(1)
        return result


class ActionDiT(nn.Module):
    """
    This is the entire DiT to predict actions
    """

    def __init__(
        self,
        obs_embed_dim: int,
        action_dim: int,
        action_len: int = 16,
        embed_dim: int = 768,
        timestep_embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.obs_embed_dim = obs_embed_dim
        self.action_dim = action_dim
        self.timestep_embed_dim = timestep_embed_dim

        # Action encoder and decoder
        # (MLP's to convert actions to tokens and decode)
        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        total_len = action_len
        self.pos_embed = nn.Parameter(
            torch.empty(1, total_len, embed_dim).normal_(std=0.02)
        )
        cond_dim = obs_embed_dim + timestep_embed_dim
        self.cond_dim = cond_dim

        self.dit_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.dit_blocks.append(
                DiTBuildingBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    use_rms_norm=use_rms_norm,
                )
            )

        # Final Adaptive Layer Norm Layer
        self.head = FinalLayer(
            dim=embed_dim, cond_dim=cond_dim, use_rms_norm=use_rms_norm
        )

        # Adaptive Layer Norm specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        def init_linear(m, std=None, val=None):
            if isinstance(m, nn.Linear):
                (
                    nn.init.normal_(m.weight, std=std)
                    if std
                    else (
                        nn.init.constant_(m.weight, val)
                        if val is not None
                        else nn.init.xavier_uniform_(m.weight)
                    )
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_linear)
        [init_linear(m, std=0.02) for m in self.timestep_embedding]
        [
            [init_linear(m, val=0) for m in b.ada_ln_modulation]
            for b in self.dit_blocks
        ]
        [
            init_linear(m, val=0)
            for m in [*self.head.ada_ln_modulation, self.head.final_linear]
        ]

    def forward(
        self,
        obs_embed,  # Concatenatd features of shape: [B, feature_dim]
        actions,  # Noised actions of shape [B, action_horizon, action_dim]
        timesteps,  # Diffusion timesteps [B, timesteps]
    ):
        # Tokenize actions
        action_embed = self.action_encoder(actions)

        # Add position embeddings to action tokens.
        x = action_embed
        x = x + self.pos_embed

        # Adaptive LayerNorm Conditioning
        # Create embeddings for timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.expand(actions.shape[0]).to(
                dtype=torch.long, device=actions.device
            )
        timesteps_emb = self.timestep_embedding(timesteps)

        # Concatenate timestep embedding + observation embeddingg
        cond = torch.cat([obs_embed, timesteps_emb], dim=-1)

        # Forward through model
        for blocks in self.dit_blocks:
            x = blocks(x, cond)

        # Final adaptive LayerNorm
        x = self.head(x, cond)

        # Decode outputs
        action_noise_pred = self.action_decoder(x)
        return action_noise_pred