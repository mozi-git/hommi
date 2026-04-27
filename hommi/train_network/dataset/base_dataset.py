from diffusion_policy.dataset.base_dataset import BaseDataset as DiffusionPolicyBaseDataset

class BaseDataset(DiffusionPolicyBaseDataset):
    def shuffle_data_ordering(self, seed:int) -> None:
        """Shuffle any internal index ordering used by the dataset."""
        raise NotImplementedError

    def requires_epoch_shuffle(self) -> bool:
        """Whether index ordering should be reshuffled between epochs."""
        raise NotImplementedError

