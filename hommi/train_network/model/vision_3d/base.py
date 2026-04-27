import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(
        self,
        frame_stack=1,
        shape_meta=None,
        **kwargs
    ):
        super().__init__()
        self.frame_stack = frame_stack
        self.shape_meta = shape_meta

        # These need to be populated by subclasses
        self.n_out_perception = 0
        self.d_out_perception = 0
        self.n_out_lowdim = 0
        self.d_out_lowdim = 0