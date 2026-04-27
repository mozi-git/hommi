import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from omegaconf import DictConfig
import torch

class ImageAugmentation:
    def __init__(self, shape_meta, transforms: DictConfig):
        self.key_to_transform = {}
        self.key_to_transform_list = {}

        for obs_key in shape_meta['obs']:
            # if shape_meta['obs'][obs_key]['ignore_by_policy']:
            #     continue
            if shape_meta['obs'][obs_key]['type'] == 'rgb':
                img_shape = shape_meta['obs'][obs_key]['shape'] # (C, H, W)
                img_size = img_shape[1]

                # compute the transforms for this RGB key
                transforms_list = []
                if obs_key not in transforms:
                    continue
                for transform in transforms[obs_key]:
                    if type(transform) == DictConfig:
                        if transform['type'] == 'RandomCrop':
                            ratio = transform.ratio
                            transforms_list.extend([
                                torchvision.transforms.RandomCrop(size=int(img_size * ratio)),
                                torchvision.transforms.Resize(size=img_size, antialias=True)
                            ])
                        elif transform['type'] == 'CenterCrop':
                            ratio = transform.ratio
                            transforms_list.extend([
                                torchvision.transforms.CenterCrop(size=int(img_size * ratio)),
                                torchvision.transforms.Resize(size=img_size, antialias=True)
                            ])
                        else:
                            raise ValueError(f"Unsupported transform type: {transform['type']}")
                    else:
                        transforms_list.append(transform)

                self.key_to_transform[obs_key] = torch.nn.Sequential(*transforms_list)
                self.key_to_transform_list[obs_key] = transforms_list

    def get_transform(self, key) -> torchvision.transforms.Compose:
        return self.key_to_transform[key]

    def apply_shared(self, key, img: torch.Tensor, *others: torch.Tensor):
        transforms_list = self.key_to_transform_list.get(key)
        if not transforms_list:
            return (img,) + others

        out_img = img
        out_others = list(others)
        for transform in transforms_list:
            if isinstance(transform, torchvision.transforms.RandomCrop):
                out_img, out_others = self._apply_random_crop(out_img, out_others, transform.size)
            elif isinstance(transform, torchvision.transforms.CenterCrop):
                out_img = TF.center_crop(out_img, transform.size)
                out_others = [TF.center_crop(other, transform.size) for other in out_others]
            elif isinstance(transform, torchvision.transforms.Resize):
                out_img = TF.resize(out_img, transform.size, antialias=transform.antialias)
                out_others = [
                    TF.resize(
                        other,
                        transform.size,
                        interpolation=InterpolationMode.NEAREST,
                        antialias=False
                    )
                    for other in out_others
                ]
            else:
                raise ValueError(f"Unsupported transform type: {type(transform)}")

        return (out_img,) + tuple(out_others)

    @staticmethod
    def _apply_random_crop(
        img: torch.Tensor,
        others,
        size,
    ):
        if img.ndim == 3:
            params = torchvision.transforms.RandomCrop.get_params(img, output_size=size)
            img = TF.crop(img, *params)
            others = [TF.crop(other, *params) for other in others]
            return img, others

        crops = []
        others_crops = [[] for _ in others]
        for idx in range(img.shape[0]):
            params = torchvision.transforms.RandomCrop.get_params(img[idx], output_size=size)
            crops.append(TF.crop(img[idx], *params))
            for other_idx, other in enumerate(others):
                others_crops[other_idx].append(TF.crop(other[idx], *params))
        img = torch.stack(crops, dim=0)
        others = [torch.stack(crop_list, dim=0) for crop_list in others_crops]
        return img, others
