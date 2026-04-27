"""Simple wrapper about DiffusionUnetTimmPolicy to make it compatible with hommi BasePolicy."""

import torch
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy as DiffusionUnetTimmPolicyBase
from hommi.train_network.model.common.base_policy import BasePolicy

class DiffusionUnetTimmPolicy(DiffusionUnetTimmPolicyBase, BasePolicy):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)
    
    def get_optimizer(self, lr, **kwargs):
        # set vision encoder to have lower learning rate if pretrained
        obs_encorder_lr = lr
        if self.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
        obs_encorder_params = list()
        for param in self.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        param_groups = [
            {'params': self.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=lr, **kwargs)
        return optimizer
