"""
Usage:
Training:
python train.py --config-name=train_policy_workspace
"""

import pathlib
import sys

# Allow running this file directly via `python hommi/train_network/train.py`.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# UMI is not installed as a package; add it so `diffusion_policy` imports resolve.
UMI_ROOT = REPO_ROOT / "deps" / "universal_manipulation_interface"
if str(UMI_ROOT) not in sys.path:
    sys.path.insert(0, str(UMI_ROOT))

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
