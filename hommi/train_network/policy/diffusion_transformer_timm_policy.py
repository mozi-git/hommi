"""Simple wrapper about DiffusionTransformerTimmPolicy to make it compatible with hommi BasePolicy."""

from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy as DiffusionTransformerTimmPolicyBase
from hommi.train_network.model.common.base_policy import BasePolicy

class DiffusionTransformerTimmPolicy(DiffusionTransformerTimmPolicyBase, BasePolicy):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)
