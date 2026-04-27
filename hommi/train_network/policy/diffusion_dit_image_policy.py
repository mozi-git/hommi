from typing import Dict, Optional, Tuple

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
import torch
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer
from hommi.train_network.model.diffusion.dit import ActionDiT
from hommi.train_network.model.common.base_policy import BasePolicy

class DiffusionDiTImagePolicy(BasePolicy):
    def __init__(
        self,
        name: str,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        use_flow_matching=False,
        fm_tsampler="uniform",
        num_inference_steps=None,
        obs_as_global_cond=True,
        train_diffusion_n_samples=1,
        attention_embed_dim=768,
        diffusion_timestep_embed_dim=256,
        depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_rms_norm=False,
        input_perturbation=0,
        skip_model_side_normalization: bool = False,
    ):
        """
        Args:
            skip_model_side_normalization:
                Set True to skip applying normalization on the model side.
        """
        super().__init__()
        self.name = name
        self.use_flow_matching = use_flow_matching
        self.fm_tsampler = fm_tsampler
        if self.fm_tsampler == "beta":
            self.tsampler = torch.distributions.beta.Beta(1.5, 1.0)
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()["features"][0]

        # create diffusion model
        self.input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            self.input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ActionDiT(
            obs_embed_dim=global_cond_dim,
            action_dim=action_dim,
            action_len=horizon,
            embed_dim=attention_embed_dim,
            timestep_embed_dim=diffusion_timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            use_rms_norm=use_rms_norm,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = ModuleAttrMixin()
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.obs_history = n_obs_steps
        self.n_obs_steps = (
            n_obs_steps 
        )
        self.obs_as_global_cond = obs_as_global_cond

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.input_pertub = input_perturbation
        self.train_diffusion_n_samples = train_diffusion_n_samples
        self._skip_model_side_normalization = skip_model_side_normalization

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if hasattr(self.obs_encoder, "set_normalizer"):
            self.obs_encoder.set_normalizer(self.normalizer)

    def maybe_normalize(
        self,
        data: Dict,
        normalizer_key: Optional[str] = None,
    ):
        if self._skip_model_side_normalization:
            return data
        else:
            if normalizer_key:
                return self.normalizer[normalizer_key].normalize(data)
            else:
                return self.normalizer.normalize(data)

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch

        valid_action_mask = batch.get("valid_action_mask", None)
        nactions = self.maybe_normalize(
            batch["action"],
            normalizer_key="action",
        )
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        # Normalize observation data.
        nobs = self.maybe_normalize(batch["obs"])

        # Determine the number of timesteps we are going to need.
        assert horizon == self.horizon
        num_steps = self.obs_history

        # reshape B, To, ... to B*To, where "To" in this case is
        # self.obs_history.
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :num_steps, ...].reshape(-1, *x.shape[2:])
        )

        # Encode observation data and retrieve resulting features.
        nobs_features_dict = self.obs_encoder(this_nobs)
        nobs_features = nobs_features_dict["features"]

        # reshape back to [B, To*D] where
        # D is the dimension of all the concatted features
        nobs_features = nobs_features.reshape(batch_size, -1)
        global_cond = nobs_features

        # train on multiple diffusion samples per obs, basically sampling
        # multiple noise levels.
        # NOTE: This increases the 'effective' batch-size, so batch_size != bsz
        if self.train_diffusion_n_samples != 1:
            assert self.obs_as_global_cond

            def _repeat(x):
                return torch.repeat_interleave(
                    x, repeats=self.train_diffusion_n_samples, dim=0
                )

            global_cond = _repeat(global_cond)
            trajectory = _repeat(trajectory)
            cond_data = _repeat(cond_data)
            if valid_action_mask is not None:
                valid_action_mask = _repeat(valid_action_mask)

        # This is the (potentially) repeated batch size using
        # `train_diffusion_n_samples`.
        B_repeated = trajectory.shape[0]

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        if self.input_pertub != 0:
            noise = noise + self.input_pertub * torch.randn(
                trajectory.shape, device=trajectory.device
            )

        if self.use_flow_matching:
            # Sample a random timestep for each image
            if self.fm_tsampler == "uniform":
                timestamps = torch.rand(
                    (B_repeated,), device=trajectory.device
                )
            elif self.fm_tsampler == "beta":
                timestamps = self.tsampler.sample((B_repeated,))
                timestamps = timestamps.to(trajectory.device)
            else:
                raise ValueError(f"Invalid tsampler: {self.fm_tsampler}")
            cont_t = timestamps.view(-1, *([1] * (noise.dim() - 1)))
            timesteps = (
                timestamps * self.noise_scheduler.config.num_train_timesteps
            ).long()

            # Flow step: x0 -> x1
            x0, x1 = trajectory, noise
            direction = x1 - x0
            noisy_trajectory = x0 + cont_t * direction
            # Predict the direction
            pred = self.model(
                obs_embed=global_cond,
                actions=noisy_trajectory,
                timesteps=timesteps,
            )
            target = direction
        else:
            # This is the original diffusion policy training objective
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B_repeated,),
                device=trajectory.device,
            ).long()
            # Add noise to the clean images according to the noise magnitude at
            # each timestep (this is the forward diffusion process)
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timesteps
            )
            # Predict the noise residual
            # local conditioning is never used, ignore.
            pred = self.model(
                obs_embed=global_cond,
                actions=noisy_trajectory,
                timesteps=timesteps,
            )
            # We always use epsilon in practice.
            target = noise

        loss = F.mse_loss(pred, target, reduction="none")
        # dont add loss for fake actions.
        loss_mask = torch.ones(
            trajectory.shape, device=trajectory.device, dtype=torch.bool
        )
        if valid_action_mask is not None:
            loss_mask = torch.logical_and(loss_mask, valid_action_mask)
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
    ):
        assert not condition_mask.any()
        assert local_cond is None
        assert global_cond is not None

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        if self.use_flow_matching:
            # This is the original flow matching inference
            # using Euler stepping
            timesteps = torch.linspace(
                1,
                0,
                self.num_inference_steps + 1,
                device=condition_data.device,
            )[:-1]
            timesteps = (
                timesteps * scheduler.config.num_train_timesteps
            ).long()

            for t in timesteps:
                # 1. predict model output
                t = t * torch.ones(len(trajectory)).to(trajectory.device)
                model_output = model(
                    obs_embed=global_cond,
                    actions=trajectory,
                    timesteps=t,
                )

                # 2. compute previous image: x_t -> x_t-1
                trajectory = (
                    trajectory - model_output / self.num_inference_steps
                )
        else:
            # This is the original diffusion policy inference
            # via DDPM/DDIM

            # set timesteps
            scheduler.set_timesteps(self.num_inference_steps)

            for t in scheduler.timesteps:
                # 1. predict model output
                model_output = model(
                    obs_embed=global_cond,
                    actions=trajectory,
                    timesteps=t,
                )

                # 2. compute previous image: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output, t, trajectory, generator=generator
                ).prev_sample

        return trajectory

    # This block is currently only called during the validation loss
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # Maybe normalize input.
        nobs = self.maybe_normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.obs_history

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        global_cond = None

        # Reshape for timesteps
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )

        nobs_features_dict = self.obs_encoder(this_nobs)
        nobs_features = nobs_features_dict["features"]

        # reshape back to (B, Do)
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if hasattr(self.obs_encoder, "set_normalizer"):
            self.obs_encoder.set_normalizer(self.normalizer)

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = [
            {
                "params": [param for pn, param in self.model.named_parameters()],
                "weight_decay": weight_decay,
            },
        ]
        backbone_params = list()
        other_obs_params = list()
        for key, value in self.obs_encoder.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr # for fine tuning
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer
    
    def forward(self, batch):
        return self.compute_loss(batch)
