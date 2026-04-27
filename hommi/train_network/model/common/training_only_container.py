import torch


class TrainingOnlyWrapper(torch.nn.Module):
    def __init__(self, module):
        """
        This wraps `module` s.t. y = module(x) during training, and y = x
        during inference. This is intended to wrap augmentation modules during
        training time.
        """

        super().__init__()
        self.module = module

    def forward(self, x):
        if self.training:
            return self.module(x)
        return x
