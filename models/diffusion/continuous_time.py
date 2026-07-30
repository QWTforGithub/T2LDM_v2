import math
from functools import partial
from typing import List, Literal

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm
from torch import Tensor
from utils import common

from . import base


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def log_snr_schedule_linear(t: torch.Tensor) -> torch.Tensor:
    return -log(expm1(1e-4 + 10 * (t**2)))


def log_snr_schedule_cosine(
    t: torch.Tensor,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def log_snr_schedule_cosine_shifted(
    t: torch.Tensor,
    image_d: float,
    noise_d: float,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    log_snr = log_snr_schedule_cosine(t, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    shift = 2 * math.log(noise_d / image_d)
    return log_snr + shift


def log_snr_schedule_cosine_interpolated(
    t: torch.Tensor,
    image_d: float,
    noise_d_low: float,
    noise_d_high: float,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    logsnr_low = log_snr_schedule_cosine_shifted(
        t, image_d, noise_d_low, logsnr_min, logsnr_max
    )
    logsnr_high = log_snr_schedule_cosine_shifted(
        t, image_d, noise_d_high, logsnr_min, logsnr_max
    )
    return t * logsnr_low + (1 - t) * logsnr_high


class ContinuousTimeGaussianDiffusion(base.GaussianDiffusion):
    """
    Continuous-time Gaussian diffusion
    https://arxiv.org/pdf/2107.00630.pdf
    """

    def __init__(
        self,
        denoiser: nn.Module,
        criterion: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        objective: Literal["eps", "v", "x0"] = "eps",
        prediction: Literal["eps", "v", "x0"] = "eps",
        beta_schedule: Literal[
            "linear", "cosine", "cosine_shifted", "cosine_interpolated"
        ] = "cosine",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
        image_d: float = None,
        noise_d_low: float = None,
        noise_d_high: float = None,
        num_training_steps: int = 1000,
        classifier_free_scale: float = 7.5,
        sampling: str= "ddpm",
        use_seg: bool = False,
        use_guidence_net: bool = False,
        guidence_step_interval: int = -1,
        guidence_weights: tuple = None,
        use_control_net = False,
        upsampling = False,
        up_rate=2,
        base_up_rate=0.25,
        downsampling =False,
        down_rate=0.5,
        base_down_rate=1.0,
        interpolate_rate=0.5,
        inference=False
    ):
        super().__init__(
            denoiser=denoiser,
            sampling=sampling,
            criterion=criterion,
            num_training_steps=num_training_steps,
            objective=objective,
            prediction=prediction,
            beta_schedule=beta_schedule,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            sampling_resolution=sampling_resolution,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            classifier_free_scale=classifier_free_scale,
            use_seg=use_seg,
            use_guidence_net=use_guidence_net,
            guidence_step_interval=guidence_step_interval,
            guidence_weights=guidence_weights,
            use_control_net=use_control_net,
            downsampling=downsampling,
            up_rate=up_rate,
            base_up_rate=base_up_rate,
            upsampling=upsampling,
            down_rate=down_rate,
            base_down_rate=base_down_rate,
            interpolate_rate=interpolate_rate,
            inference=inference
        )
        self.image_d = image_d
        self.noise_d_low = noise_d_low
        self.noise_d_high = noise_d_high

    def setup_parameters(self) -> None:
        if self.beta_schedule == "linear":
            self.log_snr = log_snr_schedule_linear
        elif self.beta_schedule == "cosine":
            self.log_snr = log_snr_schedule_cosine
        elif self.beta_schedule == "cosine_shifted":
            assert self.image_d is not None and self.noise_d_low is not None
            self.log_snr = partial(
                log_snr_schedule_cosine_shifted,
                image_d=self.image_d,
                noise_d=self.noise_d_low,
            )
        elif self.beta_schedule == "cosine_interpolated":
            assert (
                self.image_d is not None
                and self.noise_d_low is not None
                and self.noise_d_high is not None
            )
            self.log_snr = partial(
                log_snr_schedule_cosine_interpolated,
                image_d=self.image_d,
                noise_d_low=self.noise_d_low,
                noise_d_high=self.noise_d_high,
            )
        else:
            raise ValueError(f"invalid beta schedule: {self.beta_schedule}")

    @staticmethod
    def log_snr_to_alpha_sigma(log_snr):
        alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
        return alpha, sigma

    def get_target(self, x_0, step_t, noise):
        if self.objective == "eps":
            target = noise
        elif self.objective == "x0":
            target = x_0
        elif self.objective == "v":
            log_snr = self.log_snr(step_t)[:, None, None, None]
            alpha, sigma = self.log_snr_to_alpha_sigma(log_snr)
            target = alpha * noise - sigma * x_0
        elif self.objective == "x0":
            loss_weight = clipped_snr
        else:
            raise ValueError(f"invalid objective {self.objective}")
        return target

    def get_prediction(self, pred, x_t, alpha_t, sigma_t):
        """
        pred: 网络输出
        x_t: noisy input
        alpha_t, sigma_t: scalars or tensors (broadcastable)
        return: dict with keys ['eps', 'x0', 'v']
        """

        if self.prediction == "eps":
            eps = pred
            x0 = (x_t - sigma_t * eps) / alpha_t
            v = alpha_t * eps - sigma_t * x0

        elif self.prediction == "x0":
            x0 = pred
            eps = (x_t - alpha_t * x0) / sigma_t
            v = alpha_t * eps - sigma_t * x0

        elif self.prediction == "v":
            v = pred
            x0 = alpha_t * x_t - sigma_t * v
            eps = sigma_t * x_t + alpha_t * v

        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

        if(self.objective == "eps"):
            return eps
        elif(self.objective == "x0"):
            return x0
        elif(self.objective == "v"):
            return v

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # continuous timesteps
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_denoiser_condition(self, steps):
        return self.log_snr(steps)

    @autocast(enabled=False)
    def q_sample(self, x_0, step_t, noise):
        # forward diffusion process q(zt|x0) where 0<t<1
        log_snr = self.log_snr(step_t)[:, None, None, None]
        alpha, sigma = self.log_snr_to_alpha_sigma(log_snr)
        x_t = x_0 * alpha + noise * sigma
        return x_t, alpha, sigma

    def get_loss_weight(self, steps):
        log_snr = self.log_snr(steps)[:, None, None, None]
        snr = log_snr.exp()
        clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            clipped_snr.clamp_(max=self.min_snr_gamma)
        if self.objective == "eps":
            loss_weight = clipped_snr / snr
        elif self.objective == "v":
            loss_weight = clipped_snr / (snr + 1)
        else:
            raise ValueError(f"invalid objective {self.objective}")
        return loss_weight

    @torch.inference_mode()
    def p_sample(
        self,
        x_t: torch.Tensor,
        step_t: torch.Tensor,
        step_s: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
        text_features: Tensor | None = None,
        text_null_features: Tensor | None = None,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        conditional_x_0: Tensor | None = None,
        points=None,
        semantic=None

    ) -> torch.Tensor:
        # reverse diffusion process p(zs|zt) where 0<s<t<1
        log_snr_t = self.log_snr(step_t)[:, None, None, None]
        log_snr_s = self.log_snr(step_s)[:, None, None, None]
        alpha_t, sigma_t = self.log_snr_to_alpha_sigma(log_snr_t)
        alpha_s, sigma_s = self.log_snr_to_alpha_sigma(log_snr_s)

        if(self.use_guidence_net and not self.use_control_net and not self.inference):
            n_pred, g_pred, _,_ = self.denoiser(n_x=x_t, t=log_snr_t[:, 0, 0, 0], cemb=text_features, g_x=conditional_x_0)
        else:
            if(self.use_seg):
                points = semantic
            n_pred = self.denoiser(n_x=x_t, t=log_snr_t[:, 0, 0, 0],cemb=text_features, g_x=None, l_x=points)

        if(self.classifier_free_scale is not None):
            print("---- Using Classifier Free Scale ----")
            prediction_null = self.denoiser(x_t, log_snr_t[:, 0, 0, 0], text_null_features)
            # classifier-free: e_guide = e_null + s · (e_condition - e_null)
            n_pred = prediction_null + self.classifier_free_scale * (n_pred - prediction_null)

        if self.prediction == "eps":
            x_0 = (x_t - sigma_t * n_pred) / alpha_t
        elif self.prediction == "v":
            x_0 = alpha_t * x_t - sigma_t * n_pred
        elif self.prediction == "x0":
            x_0 = n_pred
        else:
            raise ValueError(f"invalid objective {self.prediction}")

        if self.clip_sample:
            x_0.clamp_(-self.clip_sample_range, self.clip_sample_range)

        var_noise = None
        if mode == "ddpm":
            c = -expm1(log_snr_t - log_snr_s)
            mean = alpha_s * (x_t * (1 - c) / alpha_t + c * x_0)
            var = sigma_s.pow(2) * c
            var_noise = self.randn_like(x_t, rng=rng)
            var_noise[step_t == 0] = 0
            x_s = mean + var.sqrt() * var_noise
        elif mode == "ddim":
            noise = (x_t - alpha_t * x_0) / sigma_t.clamp(min=1e-8)
            x_s = alpha_s * x_0 + sigma_s * noise
        else:
            raise ValueError(f"invalid mode {mode}")

        if(self.use_guidence_net and not self.use_control_net and not self.inference):
            return x_s, g_pred, var_noise
        else:
            return x_s, var_noise

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_noise: bool = False,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        text_features: Tensor | None = None,
        text_null_features: Tensor | None = None,
        conditional_x_0: Tensor | None = None,
        points=None,
        batches=None,
        semantic=None,
        sampling_condition=None,
    ):
        noise = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        x = noise
        noises = []
        points_batches = []

        if(self.upsampling):

            interpolate = []
            start = 0
            end = 0

            for batch in batches:
                end += batch
                points_batch = points[start:end, :] # [N,C]

                # ---- 下采样 ----
                points_batches.append(points_batch)
                # ---- 下采样 ----

                if (self.interpolate_rate > 0):
                    # ---- interpolate points -----
                    # ---- 插值 ----
                    i_batch = points_batch.unsqueeze(0).permute(0,2,1) # [1,C,N]
                    i_batch = common.midpoint_interpolate(i_batch, up_rate=self.up_rate) # [1,C,N/2]
                    i_batch = i_batch.permute(0,2,1).squeeze() # [1,N/2,C] -> [N/2,C]
                    # ---- 插值 ----

                    # ---- RM ----
                    i_batch = common.points_as_images_torch(i_batch, size=(self.sampling_shape[1], int(self.sampling_shape[2]))).permute(2,0,1) # [C,H,W]
                    i_batch = i_batch.unsqueeze_(0)
                    i_batch = common.convert_depth(i_batch)
                    i_batch = common.normalize(i_batch)
                    # ---- RM ----
                    interpolate.append(i_batch)
                    # ---- interpolate points -----

                start += batch

            if (self.interpolate_rate > 0):
                interpolate = torch.cat(interpolate, dim=0)

            x = self.randn(batch_size, *(self.sampling_shape[0], self.sampling_shape[1], int(self.sampling_shape[2])), rng=rng, device=self.device)

            # x[:, [0], :, :] = x[:, [0], :, :] + interpolate

        elif(self.downsampling):

            interpolate = []
            start = 0
            end = 0

            for batch in batches:
                end += batch
                points_batch = points[start:end, :]  # [N,C]

                # ---- 上采样 ----
                points_batches.append(points_batch)
                # ---- 上采样 ----

                if (self.interpolate_rate > 0):
                    # ---- 插值 ----
                    # ---- 下采样 ----
                    i_batch = points_batch.unsqueeze(0).permute(0, 2, 1)  # [1,C,N]
                    i_batch = common.midpoint_interpolate(i_batch, up_rate=self.down_rate)  # [1,C,N/2]
                    i_batch = i_batch.permute(0, 2, 1).squeeze()  # [1,N/2,C] -> [N/2,C]
                    # ---- 下采样 ----

                    # ---- RM ----
                    i_batch = common.points_as_images_torch(i_batch, size=(self.sampling_shape[1], int(self.sampling_shape[2]))).permute(2, 0, 1)  # [C,H,W]
                    i_batch = i_batch.unsqueeze_(0)
                    i_batch = common.convert_depth(i_batch)
                    i_batch = common.normalize(i_batch)
                    # ---- RM ----
                    interpolate.append(i_batch)
                    # ---- 插值 ----

                start += batch

            if(self.interpolate_rate > 0):
                interpolate = torch.cat(interpolate, dim=0)

            x = self.randn(batch_size, *(self.sampling_shape[0], self.sampling_shape[1], int(self.sampling_shape[2])), rng=rng,  device=self.device)

        noises.append(x)
        g_preds = []
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)
        for i in tqdm(range(num_steps), mininterval=0.0, **tqdm_kwargs):
            step_t = steps[:, i]
            step_s = steps[:, i + 1]
            if(self.use_guidence_net and conditional_x_0 is not None and not self.use_control_net and not self.inference):
                x, g_pred, noise = self.p_sample(
                    x,
                    step_t,
                    step_s,
                    text_features=text_features,
                    text_null_features=text_null_features,
                    rng=rng,
                    mode=mode,
                    conditional_x_0=conditional_x_0,
                )
                g_preds.append(g_pred)
            else:
                x, noise = self.p_sample(
                    x,
                    step_t,
                    step_s,
                    text_features=text_features,
                    text_null_features=text_null_features,
                    rng=rng,
                    mode=mode,
                    points=sampling_condition,
                    semantic=semantic
                )
                if((self.upsampling or self.downsampling) and self.interpolate_rate > 0):
                    x[:,[0],:,:] = (1-self.interpolate_rate) * x[:,[0],:,:] + self.interpolate_rate * interpolate

            if(return_noise): noises.append(noise)

        if(return_noise): noises = torch.stack(noises, dim=0)

        if(self.upsampling):
            return (x, points_batches, noises) if return_noise else (x, points_batches, None) # down -> up

        if(self.downsampling):
            return (x, points_batches, noises) if return_noise else (x, points_batches, None) # down -> up

        if(self.use_guidence_net and not self.use_control_net and not self.inference):
            return (x, g_pred, noises) if return_noise else (x, g_pred, None)
        else:
            return (x, noises) if return_noise else (x, None)

