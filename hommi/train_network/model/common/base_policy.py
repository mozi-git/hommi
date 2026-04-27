from typing import Dict, Optional
import torch

from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class BasePolicy(BaseImagePolicy):
    def predict_action_training(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prediction path used for training-time evaluation."""
        return self.predict_action(obs_dict)
    
    def num_available_actions(self) -> Optional[int]:
        """Returns the number of actions that are available to execute. Returns None if no limit is imposed."""
        return None
    
    # reset state for stateful policies
    def reset(self, action_exec_horizon=None):
        """The actual horizon of the action sequence to execute. This should be <= the horizon of the policy."""
        pass

    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError
