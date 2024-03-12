import copy
import functools
import os
import blobfile as bf
import torch as th
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.nn import update_ema
from diffusion.resample import LossAwareSampler, UniformSampler
from utils.common_util import draw_scalar_field2D
from utils import dist_util

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        diffusion_net,
        triplane_loss_type,
        timestep_respacing,
        training_step,
        model,
        diffusion,
        data,
        val_data,
        ssc_refine,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.triplane_loss_type = triplane_loss_type
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.ssc_refine = ssc_refine
        self.training_step = training_step
        self.timestep_respacing = timestep_respacing
        self.diffusion_net = diffusion_net
                      
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        tblog_dir = os.path.join(logger.get_current().get_dir(), "tblog")
        self.tb = SummaryWriter(tblog_dir)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0 :
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            
        if self.diffusion_net != 'unet_voxel':
            # Save the last checkpoint if it wasn't already saved.
            if (self.step - 1) % self.save_interval != 0:
                self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        
        if self.diffusion_net != 'unet_voxel':
            if self.step % self.log_interval == 0:
                self._sample_and_visualize()

    def _sample_and_visualize(self):
        print("Sampling and visualizing...")
        self.ddp_model.eval()

        batch, cond = next(self.val_data)

        _shape = [len(cond['path'])] + list(batch.shape[1:])
        with th.no_grad():
            if self.ssc_refine:
                large_T = th.tensor([self.training_step-1] * _shape[0], device=dist_util.dev())
                batch = batch.to(dist_util.dev())
                m_t = self.diffusion.q_sample(batch, large_T)
                noise = self.ddp_model(m_t, large_T, cond['H'], cond['W'], cond['D'], cond['y']).to(dist_util.dev())
            else : noise = None
            sample = self.diffusion.p_sample_loop(self.ddp_model, _shape, noise = noise, progress=True, model_kwargs=cond, clip_denoised=True)
        sample = sample.detach().cpu().numpy()
        feat_dim = sample.shape[1]
        
        for i in range(sample.shape[0]):
            for c in range(feat_dim//4):
                fig = draw_scalar_field2D(sample[i, c*4])
                self.tb.add_figure(f"sample{i}/channel{c*4}", fig, global_step=self.step)
            if self.ssc_refine :
                for c in range(feat_dim//4):
                    fig = draw_scalar_field2D(cond['y'][i, c*4].detach().cpu().numpy())
                    self.tb.add_figure(f"sample{i}/condition{c*4}", fig, global_step=self.step)
            for c in range(feat_dim//4):
                fig = draw_scalar_field2D(batch[i, c*4].detach().cpu().numpy())
                self.tb.add_figure(f"sample{i}/gt{c*4}", fig, global_step=self.step)
                       
        self.ddp_model.train()


    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch.to(dist_util.dev())
            micro_cond = {}

            for k, v in cond.items():
                if (k != 'path'):
                    micro_cond[k] = v.to(dist_util.dev())
                else :
                    micro_cond[k] = [i for i in v]
                                
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,)

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)

            if self.step % 10 == 0:
                self.log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        if self.step % 10 == 0:
            self.tb.add_scalar("step", self.step + self.resume_step, global_step=self.step)
            self.tb.add_scalar("samples", (self.step + self.resume_step + 1) * self.global_batch, global_step=self.step)
            self.tb.add_scalar("lr", self.opt.param_groups[0]["lr"], global_step=self.step)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        # save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # dist.barrier()

    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            loss_dict = {}
            logger.logkv_mean(key, values.mean().item())
            loss_dict[f"{key}_mean"] = values.mean().item()
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                loss_dict[f"{key}_q{quartile}"] = sub_loss
            self.tb.add_scalars(f"{key}", loss_dict, global_step=self.step)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("_")[-1].split(".")[0]
    return int(split)


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

