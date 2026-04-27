import os
from omegaconf import OmegaConf
import numpy as np
from .import_umi_source import get_umi_dir
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

def symlink_absolute(src, dest, **kwargs):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    os.symlink(src, dest, **kwargs)

def symlink_relative(src, dest, **kwargs):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    relative_src_path = os.path.relpath(src, start=os.path.dirname(dest))
    os.symlink(relative_src_path, dest, **kwargs)

def register_omegeconf_resolvers():
    OmegaConf.register_new_resolver('umi_dir', lambda: get_umi_dir())

def get_identity_normalizer_from_shape(shape):
    return SingleFieldLinearNormalizer.create_manual(
        scale=np.ones(shape, dtype=np.float32),
        offset=np.zeros(shape, dtype=np.float32),
        input_stats_dict={
            'min': np.zeros(shape, dtype=np.float32),
            'max': np.ones(shape, dtype=np.float32),
            'mean': np.zeros(shape, dtype=np.float32),
            'std': np.ones(shape, dtype=np.float32)
        }
    )