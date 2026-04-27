import os
import time
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import copy
import random
import pickle
import dill
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import accelerate
from accelerate import Accelerator
from datetime import timedelta
import wandb
from accelerate import InitProcessGroupKwargs
from hommi.train_network.utils.training_utils import get_gradient_norm
from hommi.train_network.dataset.base_dataset import BaseDataset
from hommi.train_network.model.common.base_policy import BasePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BasePolicy = hydra.utils.instantiate(cfg.model)

        self.ema_model: BasePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable parallelism to remove warnings about tokenizer issues after process is forked due to dataloader having multiple worker processes
        timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=120))
        accelerator = Accelerator(log_with='wandb', kwargs_handlers=[timeout], mixed_precision=cfg.training.mixed_precision)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # ensure all processes use the same output directory
        output_dirs = accelerate.utils.gather_object([self.output_dir] if accelerator.is_main_process else [''])
        main_output_dir = [x for x in output_dirs if x][0]
        self._output_dir = main_output_dir
        accelerator.print(f'Started training. Run dir: {self.output_dir}')

        # configure optimizer
        for key, value in cfg.optimizer.items():
            if key == 'lr' or key == 'obs_encoder_lr':
                cfg.optimizer[key] *= accelerator.num_processes # scale LR by num GPUs; see see https://huggingface.co/docs/accelerate/concept_guides/performance

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.num_grad_steps = 0
        self.epoch = 0

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                try:
                    self.load_checkpoint(path=lastest_ckpt_path)
                except RuntimeError as exc:
                    accelerator.print(
                        f"Checkpoint load failed with strict=True ({exc}). "
                        "Retrying with strict=False to allow normalizer key mismatches."
                    )
                    payload = torch.load(
                        lastest_ckpt_path.open('rb'),
                        pickle_module=dill
                    )
                    self.load_payload(payload, strict=False)

        # configure train and validation datasets
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        assert isinstance(dataset, BaseDataset)
        val_dataset: BaseDataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        accelerator.print('train dataset length:', len(dataset), 'dataloader length:', len(train_dataloader))
        accelerator.print('val dataset length:', len(val_dataset), 'dataloader length:', len(val_dataloader))

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure checkpoint and make sure the monitor_key is valid
        old_monitor_key = cfg.checkpoint.topk.monitor_key
        old_mode = cfg.checkpoint.topk.mode
        if cfg.checkpoint.topk.monitor_key == 'train_mean_score' or cfg.checkpoint.topk.monitor_key == 'test_mean_score':
            # Rollout metrics are not used in this training workspace.
            split = 'val' if len(val_dataloader) > 0 else 'train'
            cfg.checkpoint.topk.monitor_key = f'{split}_action_mse_error'
            cfg.checkpoint.topk.mode = 'min'
        if 'action_mse_error' in cfg.checkpoint.topk.monitor_key:
            if cfg.training.sample_every == -1:
                # invalid, try val loss
                cfg.checkpoint.topk.monitor_key = 'val_loss'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use sample every
                checkpoint_every = cfg.training.sample_every
        if cfg.checkpoint.topk.monitor_key == 'val_loss':
            if cfg.training.val_every == -1:
                # invalid, try train loss
                cfg.checkpoint.topk.monitor_key = 'train_loss'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use val every
                checkpoint_every = cfg.training.val_every
        if cfg.checkpoint.topk.monitor_key == 'train_loss':
            # train_loss is always a valid metric to checkpoint on
            checkpoint_every = 1

        if cfg.checkpoint.topk.monitor_key != old_monitor_key:
            print(f'NOTE: updating checkpoint monitor key from `{old_monitor_key}` ({old_mode}) to `{cfg.checkpoint.topk.monitor_key}` ({cfg.checkpoint.topk.mode})')
            cfg.checkpoint.topk.format_str = cfg.checkpoint.topk.format_str.replace(old_monitor_key, cfg.checkpoint.topk.monitor_key)
        
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        ) if cfg.checkpoint.topk.monitor_key != 'train_loss' else None
        topk_manager_train_loss = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            monitor_key='train_loss',
            mode='min',
            k=cfg.checkpoint.topk.k,
            format_str='epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt'
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.training.visualize_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                accelerator.wait_for_everyone() # since some of the evaluations can take a while (env rollout happens only on main process), halt all processes until it's finished to prevent timeout

                self.model.train()
                if self.ema_model is not None:
                    self.ema_model.train()

                if dataset.requires_epoch_shuffle() and self.epoch != 0:
                    # Shuffle data ordering and recreate dataloaders when dataset indices change.
                    dataset.shuffle_data_ordering(seed=self.epoch)
                    val_dataset.shuffle_data_ordering(seed=self.epoch)
                    train_dataloader = DataLoader(dataset, **cfg.dataloader)
                    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
                    train_dataloader, val_dataloader = accelerator.prepare(
                        train_dataloader, val_dataloader
                    )
                    accelerator.print(f'Epoch {local_epoch_idx}. Recreated dataloaders')
                    accelerator.print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
                    accelerator.print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                step_log = dict()
                train_losses = list()
                timing_threshold_sec = getattr(cfg.training, 'timing_log_threshold_sec', 5.0)
                prev_step_end = time.perf_counter()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        dataloader_wait_sec = time.perf_counter() - prev_step_end
                        if dataloader_wait_sec > timing_threshold_sec:
                            accelerator.print(
                                f"[timing] dataloader_wait={dataloader_wait_sec:.2f}s "
                                f"(epoch={self.epoch} batch={batch_idx})"
                            )
                        step_start = time.perf_counter()
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # always use the latest batch
                        train_sampling_batch = batch
                        step_log = {}

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # log grad norms
                        if cfg.training.log_grad_norm_every != -1 and self.num_grad_steps % cfg.training.log_grad_norm_every == 0 and self.global_step % cfg.training.gradient_accumulate_every == 0:
                            step_log['grad_norm'] = get_gradient_norm(self.model)
                            if cfg.training.clip_grad_norm:
                                step_log['grad_norm_clipped'] = min(cfg.training.clip_grad_norm, step_log['grad_norm'])

                        if cfg.training.clip_grad_norm and accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(self.model.parameters(), cfg.training.clip_grad_norm)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.num_grad_steps += 1
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log.update({
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'grad_steps': self.num_grad_steps,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        })

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                        
                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                        step_duration_sec = time.perf_counter() - step_start
                        if step_duration_sec > timing_threshold_sec:
                            accelerator.print(
                                f"[timing] step_compute={step_duration_sec:.2f}s "
                                f"(epoch={self.epoch} batch={batch_idx})"
                            )
                        prev_step_end = time.perf_counter()

                if len(train_losses) > 0:
                    del raw_loss, loss, batch
                
                # at the end of each epoch log epoch average train loss
                train_loss_average = np.mean(train_losses)
                step_log['train_loss_epoch_average'] = step_log['train_loss'] = train_loss_average # also sets 'train_loss' to epoch average in case we checkpoint based on 'train_loss' we should be using the epoch average not just one batch

                # ========= eval for this epoch ==========
                self.optimizer.zero_grad() # ensure optimizer is not holding any gradients so that we free up GPU memory for rollout (we need this since we might not end on a gradient accumulation step where we step the model and then zero grads); this is here in the case that the final step is not a gradient update step (due to multiple steps of gradient accumulation)
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                # if cfg.training.val_every != -1 and self.epoch % cfg.training.val_every == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                #     with torch.inference_mode():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 loss = policy(batch)
                #                 val_losses.append(loss.item())
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             del batch
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss

                # compute action MSE error on both training and validation datasets
                def compute_action_mse(category, pred_action, gt_action):
                    log = {}

                    log[f'{category}/action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action).item()
                    action_indexing = dataset.action_indexing
                    for key, (start, end) in action_indexing.items():
                        log[f'{category}/{key}'] = torch.nn.functional.mse_loss(pred_action[..., start:end], gt_action[..., start:end]).item()

                    return log
                
                if cfg.training.sample_every != -1 and self.epoch % cfg.training.sample_every == 0 and accelerator.is_main_process:
                    with torch.inference_mode():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action_training(batch['obs'])['action_pred']
                        mse_log = compute_action_mse('train', pred_action, gt_action)
                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            visualize_every = getattr(cfg.training, 'visualize_every', -1)
                            should_visualize = (
                                visualize_every != -1
                                and (self.epoch % visualize_every) == 0
                            )
                            if should_visualize and hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'vis_attention'):
                                policy.obs_encoder.vis_attention = True
                            pred_action = policy.predict_action_training(batch['obs'])['action_pred']
                            mse_log = compute_action_mse('val', pred_action, gt_action)
                            # visualize attention
                            if should_visualize and hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'vis_attention'):
                                # try:
                                if hasattr(policy.obs_encoder, '_last_vit_viz') and policy.obs_encoder._last_vit_viz:
                                    for key, img in policy.obs_encoder._last_vit_viz.items():
                                        # log attention visualization to wandb
                                        accelerator.log({f'val/vit_attention_{key}': wandb.Image(img)}, step=self.global_step)
                                if hasattr(policy.obs_encoder, '_last_3d_viz') and policy.obs_encoder._last_3d_viz:
                                    # log attention visualization to wandb
                                    for key, img in policy.obs_encoder._last_3d_viz.items():
                                        accelerator.log({f'val/3d_attention_{key}': wandb.Image(img)}, step=self.global_step)
                                # except Exception as e:
                                #     print(f'Error visualizing attention: {e}')
                                policy.obs_encoder.vis_attention = False
                        step_log.update(mse_log)
                        del batch
                        del gt_action
                        del pred_action
                
                # checkpoint
                if accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # always save `train_loss` checkpoint
                    topk_ckpt_path = topk_manager_train_loss.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    if (self.epoch % checkpoint_every) == 0 and topk_manager is not None:
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
        accelerator.print(f'Finished training. Run dir: {self.output_dir}')
