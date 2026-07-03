from pathlib import Path
import torch
import os
from utils import common

from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from utils.lidar import LiDARUtility

from models.T2LDM_plus_plus import CircularUNet
from examples.text_clip_nuScenes.config_text_clip_nuScenes_gn import TrainingConfig

def setup_model(
    ckpt_path,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile_denoiser: bool = False,
    load_config: bool = False,
    project_dir: str = "test"
):

    if isinstance(ckpt_path, (str, Path)):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if(load_config):
        cfg = TrainingConfig(**ckpt["cfg"])
    else:
        cfg = TrainingConfig()

    in_channels = [0, 0]
    if cfg.train_depth:
        in_channels[0] = 1
    if cfg.train_reflectance:
        in_channels[1] = 1

    if cfg.model_name == "circular_unet":
        model = CircularUNet(
            in_channel=cfg.in_channel,
            out_channel=cfg.out_channel,
            control_channel=cfg.control_channel,

            n_base_channel=cfg.n_base_channel,
            n_channels=cfg.n_channels,
            n_strides=cfg.n_strides,
            n_attn_types=cfg.n_attn_types,
            n_norm_types=cfg.n_norm_types,
            n_use_norm=cfg.n_use_norm,
            n_use_res_connection=cfg.n_use_res_connection,
            n_heads=cfg.n_heads,

            n_midd_attn_type=cfg.n_midd_attn_type,
            n_midd_norm_type=cfg.n_midd_norm_type,
            n_midd_use_norm=cfg.n_midd_use_norm,
            n_midd_use_res_connection=cfg.n_midd_use_res_connection,
            n_midd_heads=cfg.n_midd_heads,
            n_midd_channels=cfg.n_midd_channels,

            g_base_channel=cfg.g_base_channel,
            g_channels=cfg.g_channels,
            g_strides=cfg.g_strides,
            g_attn_types=cfg.g_attn_types,
            g_norm_types=cfg.g_norm_types,
            g_use_norm=cfg.g_use_norm,
            g_use_res_connection=cfg.g_use_res_connection,
            g_heads=cfg.g_heads,

            g_midd_attn_type=cfg.g_midd_attn_type,
            g_midd_norm_type=cfg.g_midd_norm_type,
            g_midd_use_norm=cfg.g_midd_use_norm,
            g_midd_use_res_connection=cfg.g_midd_use_res_connection,
            g_midd_heads=cfg.g_midd_heads,
            g_midd_channels=cfg.g_midd_channels,

            dropout=cfg.dropout,
            use_circularconv_shortcut=cfg.use_circularconv_shortcut,
            skip_connection_scale=cfg.skip_connection_scale,
            freeze_guidence_net=cfg.freeze_guidence_net,
            attention_gate=cfg.attention_gate,

            resolution=cfg.resolution,
            fov=cfg.fov,

            text_channels=cfg.text_channels,
            t_channels=cfg.t_channels,

            use_text=cfg.use_text,
            use_pe=cfg.use_pe,
            use_rope=cfg.use_rope,
            use_dpe_n=cfg.use_dpe_n,
            use_dpe_g=cfg.use_dpe_g,
            use_zero_weight=cfg.use_zero_weight,
            use_guidence_net=cfg.use_guidence_net,
            use_control_net=cfg.use_control_net,
            use_image_encoder=cfg.use_image_encoder
        )
    elif cfg.model_name == "sparse_unet":
        pass
    else:
        raise ValueError(f"Unknown: {cfg.model_name}")

    model.coords = common.get_hdl64e_linear_ray_angles(resolution=cfg.resolution, fov=cfg.fov)

    if cfg.diffusion_timesteps_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            denoiser=model,
            criterion=cfg.criterion,
            num_training_steps=cfg.diffusion_num_training_steps,
            objective=cfg.diffusion_objective,
            prediction=cfg.diffusion_prediction,
            beta_schedule=cfg.diffusion_beta_schedule,
            min_snr_loss_weight=cfg.diffusion_min_snr_loss_weight,
            min_snr_gamma=cfg.diffusion_min_snr_gamma,
            clip_sample=cfg.diffusion_clip_sample,
            clip_sample_range=cfg.diffusion_clip_sample_range,
            classifier_free_scale=cfg.diffusion_classifier_free_scale,
            sampling=cfg.diffusion_sampling,
            guidence_step_interval=cfg.diffusion_guidence_step_interval,
            guidence_weights=cfg.diffusion_guidence_weights,
            use_guidence_net=cfg.use_guidence_net,
            use_control_net=cfg.use_control_net,
            upsampling=cfg.upsampling,
            up_rate=cfg.up_rate,

            downsampling=cfg.downsampling,
            down_rate=cfg.down_rate,
            use_seg=cfg.use_seg,
            interpolate_rate=cfg.interpolate_rate,
            inference=True
        )
    elif cfg.diffusion_timesteps_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            denoiser=model,
            criterion=cfg.criterion,
            objective=cfg.diffusion_objective,
            prediction=cfg.diffusion_prediction,
            beta_schedule=cfg.diffusion_beta_schedule, # "cosine"
            min_snr_loss_weight=cfg.diffusion_min_snr_loss_weight,
            min_snr_gamma=cfg.diffusion_min_snr_gamma,
            clip_sample=cfg.diffusion_clip_sample,
            clip_sample_range=cfg.diffusion_clip_sample_range,
            classifier_free_scale=cfg.diffusion_classifier_free_scale,
            sampling=cfg.diffusion_sampling,
            guidence_step_interval=cfg.diffusion_guidence_step_interval,
            guidence_weights=cfg.diffusion_guidence_weights,
            use_guidence_net=cfg.use_guidence_net,
            use_control_net=cfg.use_control_net,
            upsampling=cfg.upsampling,
            up_rate=cfg.up_rate,
            downsampling=cfg.downsampling,
            down_rate=cfg.down_rate,
            use_seg=cfg.use_seg,
            interpolate_rate=cfg.interpolate_rate,
            inference=True
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion_timesteps_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    ddpm.load_state_dict(state_dict)
    print(f"loading : {ckpt_path} ...")
    ddpm.eval()
    ddpm.to(device)

    if compile_denoiser:
        ddpm.denoiser = torch.compile(ddpm.denoiser)

    lidar_utils = LiDARUtility(
        resolution=cfg.resolution,
        image_format=cfg.image_format,
        fov=cfg.fov,
        depth_range=cfg.depth_range,
        ray_angles=ddpm.denoiser.coords,
        project_dir=os.path.join(project_dir, "plys")
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {model.resolution}",
                f"denoiser: {model.__class__.__name__}",
                f"diffusion: {ddpm.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {common.count_parameters(ddpm):,}",
            ],
            sep="\n",
        )

    return ddpm, lidar_utils, cfg


def setup_rng(seeds: list[int], device: torch.device | str):
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
