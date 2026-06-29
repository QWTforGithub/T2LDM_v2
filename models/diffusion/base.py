from typing import List, Literal
import einops
import torch
from torch import nn
from torch.cuda.amp import autocast
from utils import common
from torch.special import expm1
import math


class GaussianDiffusion(nn.Module):
    """
    Base class for continuous/discrete Gaussian diffusion models
    """

    def __init__(
            self,
            denoiser: nn.Module,
            sampling: Literal["ddpm", "ddim"] = "ddpm",
            criterion: Literal["l2", "l1", "huber"] | nn.Module = "l2",
            num_training_steps: int = 1000,
            objective: Literal["eps", "v", "x0"] = "eps",  # loss fitting
            prediction: Literal["eps", "v", "x0"] = "eps",  # net prediction
            beta_schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
            min_snr_loss_weight: bool = True,
            min_snr_gamma: float = 5.0,
            sampling_resolution: tuple[int, int] | None = None,
            clip_sample: bool = True,
            clip_sample_range: float = 1,
            classifier_free_scale: float = 7.5,
            use_seg: bool = False,
            guidence_step_interval: int = -1,
            guidence_weights: tuple = None,
            use_guidence_net: bool = False,
            use_control_net: bool = False,
            upsampling: bool = False,
            up_rate=2,
            base_up_rate=0.25,
            downsampling: bool = False,
            down_rate=0.5,
            base_down_rate=1.0,
            interpolate_rate=0.5,
            inference=False
    ):
        super().__init__()
        self.denoiser = denoiser
        self.sampling = sampling
        self.num_training_steps = num_training_steps
        self.objective = objective
        self.prediction = prediction
        self.beta_schedule = beta_schedule
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.classifier_free_scale = classifier_free_scale
        self.use_seg = use_seg
        self.use_guidence_net = use_guidence_net
        self.use_control_net = use_control_net
        self.upsampling = upsampling
        self.up_rate = up_rate
        self.base_up_rate = base_up_rate
        self.downsampling = downsampling
        self.down_rate = down_rate
        self.base_down_rate = base_down_rate
        self.interpolate_rate = interpolate_rate
        self.inference = inference

        if (use_guidence_net):
            self.guidence_weighter = common.SCRGLossWeight(
                step_interval=guidence_step_interval,
                weights=guidence_weights
            )
            self.guidence_criterion = nn.MSELoss(reduction="none")

        if criterion == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif criterion == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif criterion == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif isinstance(criterion, nn.Module):
            self.criterion = criterion
        else:
            raise ValueError(f"invalid criterion: {criterion}")
        if hasattr(self.criterion, "reduction"):
            assert self.criterion.reduction == "none"

        if sampling_resolution is None:
            assert hasattr(self.denoiser, "resolution")
            assert hasattr(self.denoiser, "in_channels")
            self.sampling_shape = (self.denoiser.in_channels, *self.denoiser.resolution)
        else:
            assert len(sampling_resolution) == 2
            assert hasattr(self.denoiser, "in_channels")
            self.sampling_shape = (self.denoiser.in_channels, *sampling_resolution)

        self.setup_parameters()
        self.register_buffer("_dummy", torch.tensor([]))

    @property
    def device(self):
        return self._dummy.device

    def randn(
            self,
            *shape,
            rng: List[torch.Generator] | torch.Generator | None = None,
            **kwargs,
    ) -> torch.Tensor:
        if rng is None:
            return torch.randn(*shape, **kwargs)
        elif isinstance(rng, torch.Generator):
            return torch.randn(*shape, generator=rng, **kwargs)
        elif isinstance(rng, list):
            assert len(rng) == shape[0]
            return torch.stack(
                [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng]
            )
        else:
            raise ValueError(f"invalid rng: {rng}")

    def randn_like(
            self,
            x: torch.Tensor,
            rng: List[torch.Generator] | torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)

    def setup_parameters(self) -> None:
        raise NotImplementedError

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def p_sample(self, *args, **kwargs):
        raise NotImplementedError

    @autocast(enabled=False)
    def q_sample(self, x_0, steps, noise):
        raise NotImplementedError

    def get_denoiser_condition(self, steps: torch.Tensor):
        raise NotImplementedError

    def get_target(self, x_0, steps, noise):
        raise NotImplementedError

    def get_prediction(self, pred, x_t, alpha_t, sigma_t):
        raise NotImplementedError

    def get_loss_weight(self, steps):
        raise NotImplementedError

    def p_loss(
            self,
            x_0: torch.Tensor,
            steps: torch.Tensor,
            loss_mask: torch.Tensor | None = None,
            print_loss: bool = False,
            text: torch.Tensor | None = None,
            semantic: torch.Tensor | None = None,
            current_steps: int = None,
            points=None, # 全部点的堆积
            sampling_condition=None,
            batches=None # 每个点云的点数量
    ) -> torch.Tensor:

        loss_weights = [1, 0.0, 0.0, 0.0, 0.0, 0.1]
        noise = self.randn_like(x_0)
        x_t, alpha_t, sigma_t = self.q_sample(x_0, steps, noise)
        t = self.get_denoiser_condition(steps)

        target = self.get_target(x_0, steps, noise)

        if (self.use_control_net):

            l_x = sampling_condition
            if (self.use_seg):
                l_x = semantic

            n_pred = self.denoiser(n_x=x_t, t=t, cemb=text, g_x=None, l_x=l_x)
            n_pred = self.get_prediction(pred=n_pred, x_t=x_t, alpha_t=alpha_t, sigma_t=sigma_t)

            n_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
            n_loss = self.criterion(n_pred, target)  # (B,C,H,W)
            n_loss = einops.reduce(n_loss * n_mask, "B ... -> B ()", "sum")
            n_mask = einops.reduce(n_mask, "B ... -> B ()", "sum")
            n_loss = n_loss / n_mask.add(1e-8)  # (B,)

            loss = ((loss_weights[0] * n_loss) * self.get_loss_weight(steps)).mean()

            if (print_loss):
                print(f"\n n_Loss : {loss.item()}")

        elif (self.use_guidence_net):

            n_pred, g_pred, n_stage_feats, g_stage_feats = self.denoiser(n_x=x_t, t=t, cemb=text, g_x=x_0)

            n_pred = self.get_prediction(pred=n_pred, x_t=x_t, alpha_t=alpha_t, sigma_t=sigma_t)
            n_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
            n_loss = self.criterion(n_pred, target)  # (B,C,H,W)
            n_loss = einops.reduce(n_loss * n_mask, "B ... -> B ()", "sum")
            n_mask = einops.reduce(n_mask, "B ... -> B ()", "sum")
            n_loss = n_loss / n_mask.add(1e-8)  # (B,)
            n_loss = ((loss_weights[0] * n_loss) * self.get_loss_weight(steps)).mean()

            g_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
            g_loss = self.guidence_criterion(g_pred, x_0)
            g_loss = einops.reduce(g_loss * g_mask, "B ... -> B ()", "sum")
            g_mask = einops.reduce(g_mask, "B ... -> B ()", "sum")
            g_loss = g_loss / g_mask.add(1e-8)  # (B,)
            g_loss = g_loss.mean()

            SCRG_losses = 0
            for g_stage_feat, n_stage_feat in zip(g_stage_feats, n_stage_feats):
                scrg_loss = common.lrepa_cosine_single(n_stage_feat, g_stage_feat)  # (B,C,H,W)
                SCRG_losses += scrg_loss.mean()

            SCRG_losses = self.guidence_weighter.get_loss_weight(current_step=current_steps) * SCRG_losses

            loss = n_loss + g_loss + SCRG_losses

            if (print_loss):
                print(f"\n n_Loss : {n_loss.item()}, g_Loss : {g_loss.item()}, stage_Loss : {SCRG_losses.item()}")

        else:
            n_pred = self.denoiser(n_x=x_t, t=t, cemb=text)
            n_pred = self.get_prediction(pred=n_pred, x_t=x_t, alpha_t=alpha_t, sigma_t=sigma_t)

            n_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
            n_loss = self.criterion(n_pred, target)  # (B,C,H,W)
            n_loss = einops.reduce(n_loss * n_mask, "B ... -> B ()", "sum")
            n_mask = einops.reduce(n_mask, "B ... -> B ()", "sum")
            n_loss = n_loss / n_mask.add(1e-8)  # (B,)

            loss = ((loss_weights[0] * n_loss) * self.get_loss_weight(steps)).mean()

            if (print_loss):
                print(f"\n n_Loss : {loss.item()}")
        return loss

    def forward(
            self,
            x_0: torch.Tensor,
            text: torch.Tensor | None = None,
            semantic: torch.Tensor | None = None,
            loss_mask: torch.Tensor | None = None,
            current_steps: int = -1,
            points=None,
            batches=None,
            sampling_condition=None,
            print_loss: bool = False
    ) -> torch.Tensor:
        steps = self.sample_timesteps(x_0.shape[0], x_0.device)
        loss = self.p_loss(
            x_0=x_0,
            steps=steps,
            loss_mask=loss_mask,
            text=text,
            semantic=semantic,
            current_steps=current_steps,
            points=points,
            batches=batches,
            sampling_condition=sampling_condition,
            print_loss=print_loss
        )
        return loss

    @torch.inference_mode()
    def sample(self, *args, **kwargs):
        raise NotImplementedError
