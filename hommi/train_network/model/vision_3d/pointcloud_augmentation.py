import torch
import torch.nn as nn
import numpy as np
import einops
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class PointCloud3DAugmentation(nn.Module):
    """
    3D data augmentation for point cloud data.
    Supports vectorized rotation, translation, scaling, and point jittering.
    """
    def __init__(
        self,
        # Rotation parameters
        enable_rotation: bool = True,
        rotation_range_x: Tuple[float, float] = (-np.pi/36, np.pi/36),  # ±5 degrees
        rotation_range_y: Tuple[float, float] = (-np.pi/36, np.pi/36),  # ±5 degrees
        rotation_range_z: Tuple[float, float] = (-np.pi/18, np.pi/18),  # ±10 degrees
        
        # Translation parameters
        enable_translation: bool = True,
        translation_range: Tuple[float, float] = (-0.05, 0.05),  # ±5cm
        
        # Scaling parameters
        enable_scaling: bool = True,
        scaling_range: Tuple[float, float] = (0.9, 1.1),  # 90%-110%
        
        # Point jittering parameters
        enable_point_jittering: bool = True,
        jitter_std: float = 0.01,  # 1cm standard deviation
        jitter_clip: float = 0.03,  # clip to 3cm
        
        # Point dropout parameters
        enable_point_dropout: bool = True,
        dropout_ratio: Tuple[float, float] = (0.0, 0.1),  # 0-10% dropout
        
        # Global parameters
        augmentation_prob: float = 0.8,  # Probability of applying augmentations
        preserve_hand_regions: bool = True,  # Don't augment near hand positions
        hand_preservation_radius: float = 0.15,  # 15cm radius around hands
    ):
        super().__init__()
        
        # Store parameters
        self.enable_rotation = enable_rotation
        self.rotation_range_x = rotation_range_x
        self.rotation_range_y = rotation_range_y
        self.rotation_range_z = rotation_range_z
        
        self.enable_translation = enable_translation
        self.translation_range = translation_range
        
        self.enable_scaling = enable_scaling
        self.scaling_range = scaling_range
        
        self.enable_point_jittering = enable_point_jittering
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip
        
        self.enable_point_dropout = enable_point_dropout
        self.dropout_ratio = dropout_ratio
        
        self.augmentation_prob = augmentation_prob
        self.preserve_hand_regions = preserve_hand_regions
        self.hand_preservation_radius = hand_preservation_radius
        
        logger.info(f"Initialized PointCloud3DAugmentation with augmentation_prob={augmentation_prob}")
    
    def forward(
        self, 
        pcd: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hand_positions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply efficient 3D data augmentation to point cloud data.
        
        Args:
            pcd: Point cloud coordinates [B, T, N, 3] (XYZ)
            mask: Valid point mask [B, T, N]
            hand_positions: Dict with 'left' and 'right' hand positions [B, T, 3]
            
        Returns:
            Augmented pcd, mask
        """
        if not self.training:
            return pcd, mask
            
        B, T, N, _ = pcd.shape
        device = pcd.device
        
        # Skip augmentation with some probability - vectorized across batch
        should_augment = torch.rand(B, T, device=device) < self.augmentation_prob
        if not should_augment.any():
            return pcd, mask
            
        augmented_pcd = pcd.clone()
        augmented_mask = mask.clone() if mask is not None else None
        
        # Create hand preservation mask if needed
        hand_preserve_mask = None
        if self.preserve_hand_regions and hand_positions is not None:
            hand_preserve_mask = self._create_hand_preservation_mask_vectorized(
                pcd, hand_positions, self.hand_preservation_radius
            )
        
        # Apply geometric transformations (vectorized)
        if self.enable_rotation or self.enable_translation or self.enable_scaling:
            transform_matrices = self._generate_transformation_matrices_vectorized(B, T, device)
            augmented_pcd = self._apply_transformation_vectorized(
                augmented_pcd, transform_matrices, should_augment, hand_preserve_mask
            )
        
        # Apply point-wise jittering (vectorized)
        if self.enable_point_jittering:
            augmented_pcd = self._apply_point_jittering_vectorized(
                augmented_pcd, should_augment, hand_preserve_mask
            )
        
        # Apply point dropout (vectorized)
        if self.enable_point_dropout and augmented_mask is not None:
            augmented_mask = self._apply_point_dropout_vectorized(
                augmented_mask, should_augment, hand_preserve_mask
            )
        
        return augmented_pcd, augmented_mask
    
    def _create_hand_preservation_mask_vectorized(
        self, 
        pcd: torch.Tensor, 
        hand_positions: Dict[str, torch.Tensor], 
        radius: float
    ) -> torch.Tensor:
        """Create mask to preserve points near hand positions - fully vectorized."""
        B, T, N, _ = pcd.shape
        device = pcd.device
        
        # Initialize preservation mask (True = preserve, False = can augment)
        preserve_mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
        
        for hand_key in ['left', 'right']:
            if hand_key in hand_positions:
                hand_pos = hand_positions[hand_key]  # [B, T, 3]
                hand_pos = hand_pos.unsqueeze(2)  # [B, T, 1, 3]
                
                # Compute distances from each point to hand - vectorized
                distances = torch.norm(pcd - hand_pos, dim=-1)  # [B, T, N]
                
                # Mark points within radius as preserved
                preserve_mask = preserve_mask | (distances < radius)
        
        return preserve_mask
    
    def _generate_transformation_matrices_vectorized(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Generate 4x4 transformation matrices for all batch items at once."""
        # Initialize identity matrices for all batch items
        transform_matrices = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        
        # Generate random parameters for all batch items at once
        if self.enable_rotation:
            # Generate rotation angles for all samples
            rx = torch.rand(B, T, device=device) * (self.rotation_range_x[1] - self.rotation_range_x[0]) + self.rotation_range_x[0]
            ry = torch.rand(B, T, device=device) * (self.rotation_range_y[1] - self.rotation_range_y[0]) + self.rotation_range_y[0]
            rz = torch.rand(B, T, device=device) * (self.rotation_range_z[1] - self.rotation_range_z[0]) + self.rotation_range_z[0]
            
            # Create rotation matrices for all samples
            cos_x, sin_x = torch.cos(rx), torch.sin(rx)
            cos_y, sin_y = torch.cos(ry), torch.sin(ry)
            cos_z, sin_z = torch.cos(rz), torch.sin(rz)
            
            # Build rotation matrices - vectorized
            # Rotation around X-axis
            Rx = torch.zeros(B, T, 4, 4, device=device)
            Rx[..., 0, 0] = 1
            Rx[..., 1, 1] = cos_x
            Rx[..., 1, 2] = -sin_x
            Rx[..., 2, 1] = sin_x
            Rx[..., 2, 2] = cos_x
            Rx[..., 3, 3] = 1
            
            # Rotation around Y-axis
            Ry = torch.zeros(B, T, 4, 4, device=device)
            Ry[..., 0, 0] = cos_y
            Ry[..., 0, 2] = sin_y
            Ry[..., 1, 1] = 1
            Ry[..., 2, 0] = -sin_y
            Ry[..., 2, 2] = cos_y
            Ry[..., 3, 3] = 1
            
            # Rotation around Z-axis
            Rz = torch.zeros(B, T, 4, 4, device=device)
            Rz[..., 0, 0] = cos_z
            Rz[..., 0, 1] = -sin_z
            Rz[..., 1, 0] = sin_z
            Rz[..., 1, 1] = cos_z
            Rz[..., 2, 2] = 1
            Rz[..., 3, 3] = 1
            
            # Combine rotations: R = Rz @ Ry @ Rx
            R = torch.matmul(torch.matmul(Rz, Ry), Rx)
            transform_matrices = torch.matmul(transform_matrices, R)
        
        if self.enable_scaling:
            # Generate scaling factors for all samples
            scales = torch.rand(B, T, device=device) * (self.scaling_range[1] - self.scaling_range[0]) + self.scaling_range[0]
            
            # Apply scaling to transformation matrices
            transform_matrices[..., 0, 0] *= scales
            transform_matrices[..., 1, 1] *= scales
            transform_matrices[..., 2, 2] *= scales
        
        if self.enable_translation:
            # Generate translation vectors for all samples
            tx = torch.rand(B, T, device=device) * (self.translation_range[1] - self.translation_range[0]) + self.translation_range[0]
            ty = torch.rand(B, T, device=device) * (self.translation_range[1] - self.translation_range[0]) + self.translation_range[0]
            tz = torch.rand(B, T, device=device) * (self.translation_range[1] - self.translation_range[0]) + self.translation_range[0]
            
            # Apply translation
            transform_matrices[..., 0, 3] += tx
            transform_matrices[..., 1, 3] += ty
            transform_matrices[..., 2, 3] += tz
        
        return transform_matrices
    
    def _apply_transformation_vectorized(
        self, 
        pcd: torch.Tensor, 
        transform_matrices: torch.Tensor,
        should_augment: torch.Tensor,
        hand_preserve_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformation matrices to point cloud - fully vectorized."""
        B, T, N, _ = pcd.shape
        device = pcd.device
        
        # Convert to homogeneous coordinates
        ones = torch.ones(B, T, N, 1, device=device)
        pcd_homo = torch.cat([pcd, ones], dim=-1)  # [B, T, N, 4]
        
        # Apply transformation: transform_matrices [B, T, 4, 4] @ pcd_homo [B, T, N, 4]
        # Use einsum for efficient batched matrix multiplication
        transformed_pcd = torch.einsum('btij,btnj->btni', transform_matrices, pcd_homo)
        
        # Convert back to 3D coordinates
        transformed_pcd = transformed_pcd[..., :3]
        
        # Apply augmentation mask (only augment where should_augment is True)
        should_augment_expanded = should_augment.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        augmented_pcd = torch.where(should_augment_expanded, transformed_pcd, pcd)
        
        # Apply hand preservation mask if provided
        if hand_preserve_mask is not None:
            # Only apply transformation where hand_preserve_mask is False (not preserved)
            augment_mask = (~hand_preserve_mask).unsqueeze(-1)  # [B, T, N, 1]
            # Combine with should_augment mask
            final_augment_mask = should_augment_expanded & augment_mask
            augmented_pcd = torch.where(final_augment_mask, transformed_pcd, pcd)
        
        return augmented_pcd
    
    def _apply_point_jittering_vectorized(
        self, 
        pcd: torch.Tensor,
        should_augment: torch.Tensor,
        hand_preserve_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply random jittering to point coordinates - fully vectorized."""
        B, T, N, _ = pcd.shape
        device = pcd.device
        
        # Generate noise for all points at once
        noise = torch.randn(B, T, N, 3, device=device) * self.jitter_std
        noise = torch.clamp(noise, -self.jitter_clip, self.jitter_clip)
        
        jittered_pcd = pcd + noise
        
        # Apply augmentation mask
        should_augment_expanded = should_augment.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        augmented_pcd = torch.where(should_augment_expanded, jittered_pcd, pcd)
        
        # Apply hand preservation mask if provided
        if hand_preserve_mask is not None:
            augment_mask = (~hand_preserve_mask).unsqueeze(-1)  # [B, T, N, 1]
            # Combine with should_augment mask
            final_augment_mask = should_augment_expanded & augment_mask
            augmented_pcd = torch.where(final_augment_mask, jittered_pcd, pcd)
        
        return augmented_pcd
    
    def _apply_point_dropout_vectorized(
        self,
        mask: torch.Tensor,
        should_augment: torch.Tensor,
        hand_preserve_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply random point dropout - fully vectorized."""
        B, T, N = mask.shape
        device = mask.device
        
        # Generate dropout ratio for each batch and time step
        dropout_ratios = torch.rand(B, T, device=device) * (self.dropout_ratio[1] - self.dropout_ratio[0]) + self.dropout_ratio[0]
        dropout_ratios = dropout_ratios.unsqueeze(-1)  # [B, T, 1]
        
        # Generate random dropout mask for all points
        dropout_probs = torch.rand(B, T, N, device=device)
        dropout_mask = dropout_probs > dropout_ratios  # True means keep the point
        
        # Apply dropout
        dropped_mask = mask & dropout_mask
        
        # Apply augmentation mask
        should_augment_expanded = should_augment.unsqueeze(-1)  # [B, T, 1]
        augmented_mask = torch.where(should_augment_expanded, dropped_mask, mask)
        
        # Preserve hand regions
        if hand_preserve_mask is not None:
            # Where hand_preserve_mask is True, use original mask
            augmented_mask = torch.where(hand_preserve_mask, mask, augmented_mask)
        
        return augmented_mask