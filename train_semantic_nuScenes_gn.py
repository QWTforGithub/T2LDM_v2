import dataclasses
import datetime
import json
import os
import warnings
import random
import einops
import matplotlib.cm as cm
import torch
import torch._dynamo
import torch.nn.functional as F
import inspect
import shutil

from accelerate import Accelerator
from ema_pytorch import EMA
from simple_parsing import ArgumentParser
from accelerate.utils import broadcast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import common
from data.kitti_360.kitti_360 import KITTI360Dataset
from data.kitti_semantic.kitti_semantic import KITTISemanticDataset
from data.nuScenes.nuScenes import NuScenesDataset
from data.conditional_x0.conditionalx0 import ConditionalX0
from pathlib import Path
from models.CLIP.clip import clip
from models.T5.T5 import t5
import utils.render
from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
    GaussianDiffusion
)
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles

from models.T2LDM_final_a import CircularUNet
from configs.config_semantic_nuScenes_gn import TrainingConfig

warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True


def train(cfg):
    # ---- Log Path ----
    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.output_dir) / f"semantic_{cfg.dataset}_gn"
    project_name = datetime.datetime.now().strftime("%Y%m]%dT%H%M%S")
    print_info = cfg.print_info
    # ---- Log Path ----

    # ---- Accelerator Config ----
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        if (print_info):
            print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )

        dest_path = project_dir / project_name
        os.makedirs(dest_path, exist_ok=True)

        ContinuousTimeGaussianDiffusion_path = inspect.getfile(ContinuousTimeGaussianDiffusion)
        ContinuousTimeGaussianDiffusion_dest_path = dest_path / ContinuousTimeGaussianDiffusion_path.split("/")[-1]
        shutil.copy(ContinuousTimeGaussianDiffusion_path, ContinuousTimeGaussianDiffusion_dest_path)
        print(f"Coping ContinuousTimeGaussianDiffusion file to {ContinuousTimeGaussianDiffusion_dest_path}")

        GaussianDiffusion_path = inspect.getfile(GaussianDiffusion)
        GaussianDiffusion_dest_path = dest_path / GaussianDiffusion_path.split("/")[-1]
        shutil.copy(GaussianDiffusion_path, GaussianDiffusion_dest_path)
        print(f"Coping GaussianDiffusion file to {GaussianDiffusion_dest_path}")

        config_path = inspect.getfile(TrainingConfig)
        config_dest_path = dest_path / config_path.split("/")[-1]
        shutil.copy(config_path, config_dest_path)
        print(f"Coping config file to {config_dest_path}")

        net_path = inspect.getfile(CircularUNet)
        net_dest_path = dest_path / net_path.split("/")[-1]
        shutil.copy(net_path, net_dest_path)
        print(f"Coping network file to {net_dest_path}")

        train_path = os.path.abspath(__file__)
        train_dest_path = dest_path / train_path.split("/")[-1]
        shutil.copy(train_path, train_dest_path)
        print(f"Coping train file to {train_dest_path}")

        print("\nAccelerator配置信息: ")
        print(accelerator.state)

    device = accelerator.device
    # ---- Accelerator Config ----

    # ---- Model Config ----
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

            print_chnanels=print_info
        )
    elif cfg.model_name == "sparse_unet":
        pass
    else:
        raise ValueError(f"Unknown: {cfg.model_name}")
    # ---- Model Config ----

    # ---- Field of View Config ----
    if "spherical" in cfg.lidar_projection:
        accelerator.print(f"set HFoV: {cfg.resolution} VFoV: {cfg.fov} linear ray angles")
        model.coords = get_hdl64e_linear_ray_angles(resolution=cfg.resolution, fov=cfg.fov)
    elif "unfolding" in cfg.lidar_projection:
        accelerator.print("set dataset ray angles")
        _coords = torch.load(f"data/{cfg.dataset}/unfolding_angles.pth")
        model.coords = F.interpolate(_coords, size=cfg.resolution, mode="nearest-exact")
    else:
        raise ValueError(f"Unknown: {cfg.lidar_projection}")
    # ---- Field of View Config ----

    if accelerator.is_main_process:
        print(f"number of parameters: {common.count_parameters(model):,}")

    # ---- DDPMs Config ----
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
            use_seg=cfg.use_seg
        )
    elif cfg.diffusion_timesteps_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            denoiser=model,
            criterion=cfg.criterion,
            objective=cfg.diffusion_objective,
            prediction=cfg.diffusion_prediction,
            beta_schedule=cfg.diffusion_beta_schedule,  # "cosine"
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
            use_seg=cfg.use_seg
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion_timesteps_type}")
    ddpm.train()
    ddpm.to(device)
    # ---- DDPMs Config ----

    # ---- Text Encoder ----
    text_encoder = None
    if (cfg.use_text):
        if (cfg.clip_mode is not None):
            text_encoder = clip.load(cfg.clip_mode, device=device)
        elif (cfg.T5_mode is not None):
            text_encoder = t5(cfg.T5_mode).to(device)
    # ---- Text Encoder ----

    # ---- EMA Config ----
    ddpm_ema = None
    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.ema_decay,
            update_every=cfg.ema_update_every,
            update_after_step=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)
    # ---- EMA Config ----

    # ---- Tool Class Config ----
    lidar_utils = LiDARUtility(
        resolution=cfg.resolution,
        image_format=cfg.image_format,
        fov=cfg.fov,
        depth_range=cfg.depth_range,
        ray_angles=ddpm.denoiser.coords,
    )
    lidar_utils.project_dir = os.path.join(project_dir, project_name, "plys")
    lidar_utils.to(device)
    # ---- Tool Class Config ----

    # ---- Param Filter ----
    if (cfg.pretrained_checkpoint_dir is not None):
        ddpm, _, _ = common.set_param_grad_by_prefix(ddpm, freeze_prefixes=cfg.filer_keys, print_info=print_info)

    if (accelerator.is_main_process):
        total_count_parameters = common.total_count_parameters(ddpm)
        count_parameters = common.count_parameters(ddpm)
        frozen_count_parameters = total_count_parameters - count_parameters

        print(f"\n---- Setting require_grad ----")
        print(f"total trainable params: {total_count_parameters:,}")
        print(f"trainable params: {count_parameters:,}")
        print(f"frozen params: {frozen_count_parameters:,}")
        print("---- Setting require_grad ----\n")
    # ---- Param Filter ----

    # ---- Training Dataloader Config ----
    dataset = None
    if (cfg.dataset == "kitti_360"):
        dataset = KITTI360Dataset(
            data_root=cfg.data_root,
            training=True,
            aug=cfg.aug,
            resolution=cfg.resolution,
            depth_range=cfg.depth_range,
            fov=cfg.fov,
            print_info=print_info
        )
    elif (cfg.dataset == "kitti_semantic"):
        dataset = KITTISemanticDataset(
            data_root=cfg.data_root,
            training=True,
            aug=cfg.aug,
            resolution=cfg.resolution,
            depth_range=cfg.depth_range,
            fov=cfg.fov,
            semantic_class_num=cfg.semantic_class_num,
            print_info=print_info
        )
    elif (cfg.dataset == "nuScenes"):
        dataset = NuScenesDataset(
            data_root=cfg.data_root,
            pkl=cfg.pkl,
            training=True,
            aug=cfg.aug,
            resolution=cfg.resolution,
            depth_range=cfg.depth_range,
            fov=cfg.fov,
            text_keys=cfg.text_keys,
            semantic_class_num=cfg.semantic_class_num,
            sampling=True if (cfg.upsampling or cfg.downsampling) else False,
            print_info=print_info
        )
    else:
        pass

    if accelerator.is_main_process:
        print(f" ---- {cfg.dataset} Dataset with {len(dataset)} ---- ")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=common.collate_fn
    )
    # ---- Training Dataloader Config ----

    # ---- Validation Dataloader Config ----
    condition_guide_dataset = None
    if (cfg.use_guidence_net or cfg.use_control_net or cfg.use_text):
        condition_guide_dataset = ConditionalX0(
            conditionalx0_lidar_path=dataset.conditionalx0_lidar_path,
            conditionalx0_lidar_description=dataset.conditionalx0_lidar_description,
            conditionalx0_lidar_semantic=dataset.conditionalx0_lidar_semantic,
            conditionalx0_lidar_info=dataset.conditionalx0_lidar_info,

            semantic_class_num=cfg.semantic_class_num,

            training=True,
            aug=cfg.aug,

            resolution=cfg.resolution,
            depth_range=cfg.depth_range,
            fov=cfg.fov,

            sampling=True if cfg.upsampling or cfg.downsampling else False,

            type=cfg.dataset,
            print_info=print_info
        )

    if ((cfg.use_guidence_net or cfg.use_text or cfg.use_control_net) and accelerator.is_main_process):
        print(f" ---- ConditionalX0 {cfg.dataset} Dataset with {len(condition_guide_dataset)} ---- \n")

    condition_guide_dataloader = None
    if (cfg.use_guidence_net or cfg.use_control_net or cfg.use_text):
        condition_guide_dataloader = DataLoader(
            condition_guide_dataset,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=common.collate_fn
        )
    # ---- Validation Dataloader Config ----

    # ---- Loading Pretrained Model (For ControlNet) ----
    global_step = 0
    if cfg.pretrained_checkpoint_dir is not None:
        if accelerator.is_main_process:
            print(f"[Rank0] Loading pretrained checkpoint from {cfg.pretrained_checkpoint_dir} ...")
            _, _ = common.load_checkpoint(
                checkpoint_path=cfg.pretrained_checkpoint_dir,
                ema_model=ddpm_ema,  # rank0 有 ema_model
                strict=False,
                print_info=print_info
            )
        # ✅ barrier 同步
        accelerator.wait_for_everyone()

        # ✅ 让 rank0 的权重广播到所有 rank
        with torch.no_grad():
            for name, param in ddpm.named_parameters():
                broadcast(param, from_process=0)  # from rank0 → all ranks

        accelerator.wait_for_everyone()
        print(f"[Rank{accelerator.process_index}] Checkpoint sync done.")
    # ---- Loading Pretrained Model (For ControlNet) ----

    # ---- Optimizer and Scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in ddpm.parameters() if p.requires_grad],  # ddpm.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    lr_scheduler = common.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.num_steps * cfg.gradient_accumulation_steps,
    )
    # ---- Optimizer and Scheduler ----

    # ---- Loading Checkpoint ----
    if cfg.checkpoint_dir is not None:
        if accelerator.is_main_process:
            print(f"[Rank0] Loading pretrained checkpoint from {cfg.checkpoint_dir} ...")
            _, global_step = common.load_checkpoint(
                checkpoint_path=cfg.checkpoint_dir,
                ema_model=ddpm_ema,  # rank0 有 ema_model
                strict=True,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                print_info=print_info
            )
        # ✅ barrier 同步
        accelerator.wait_for_everyone()

        # ✅ 让 rank0 的权重广播到所有 rank
        with torch.no_grad():
            for name, param in ddpm.named_parameters():
                broadcast(param, from_process=0)  # from rank0 → all ranks

        accelerator.wait_for_everyone()
        print(f"[Rank{accelerator.process_index}] Checkpoint sync done.")
    # ---- Loading Checkpoint ----

    # ---- Put Model, Optimizer, Dataloder and Scheduler in Accelerator ----
    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

    # ---- Put Model, Optimizer, Dataloder and Scheduler in Accelerator ----

    # ==== Progress Bar ====
    progress_bar = tqdm(
        range(cfg.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
        initial=global_step
    )
    # ==== Progress Bar ====

    while global_step < cfg.num_steps:
        ddpm.train()
        for batch in dataloader:

            # ==== Data Process ====
            inputs = common.preprocess(
                inputs=common.get_inputs(),
                batch=batch,

                classifier_dropout=cfg.diffusion_classifier_dropout,
                use_text=cfg.use_text,
                use_seg=cfg.use_seg,
                train_depth=cfg.train_depth,
                train_reflectance=cfg.train_reflectance,
                upsampling=cfg.upsampling,
                downsampling=cfg.downsampling,

                resolution=cfg.resolution,
                lidar_utils=lidar_utils,
                device=device
            )
            # ==== Data Process ====

            # ==== Text Encoder ====
            if(cfg.use_text):
                text_emb = text_encoder.tokenize(inputs["cfg_texts"], device=device)
                with torch.no_grad():
                    inputs["text_features"] = text_encoder.encode_text(text_emb, pool_features=cfg.clip_pool_features) # B, 512
            # ==== Text Encoder ====

            with accelerator.accumulate(ddpm):

                # ==== Loss ====
                loss = ddpm(
                    x_0=inputs["x"],
                    text=inputs["text_features"],
                    semantic=inputs["semantic"],
                    points=inputs["points"],
                    batches=inputs["batch"],
                    sampling_condition=inputs["sampling_depth"],
                    current_steps=global_step,
                    print_loss=print_info
                )
                accelerator.backward(loss)
                # ==== Loss ====

                # ==== Check the BP of Each Param ====
                for name, param in ddpm.named_parameters():
                    if(param.grad is None and common.filer_name_keys(name=name, keys=cfg.filer_keys)):
                        print(name)
                # ==== Check the BP of Each Param ====

                # ==== Gredient CLIP ====
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                # ==== Gredient CLIP ====

                # ==== Updating ====
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # ==== Updating ====

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()
                if global_step % cfg.save_sample_steps == 0:
                    ddpm_ema.ema_model.eval()

                    # ===== Validate DDPMs ====
                    c_inputs = common.get_inputs()
                    if(cfg.use_guidence_net or cfg.use_control_net or cfg.use_text):
                        for batch in condition_guide_dataloader:
                            c_inputs = common.preprocess(
                                inputs=c_inputs,
                                batch=batch,

                                classifier_dropout=cfg.diffusion_classifier_dropout,
                                use_text=cfg.use_text,
                                use_seg=cfg.use_seg,
                                train_depth=cfg.train_depth,
                                train_reflectance=cfg.train_reflectance,
                                upsampling=cfg.upsampling,
                                downsampling=cfg.downsampling,

                                resolution=cfg.resolution,
                                lidar_utils=lidar_utils,
                                device=device
                            )

                            if (cfg.use_text):
                                text_null = [""] * len(c_inputs["texts"])
                                c_inputs["text_features"], c_inputs["text_null_features"] = common.get_text_features(
                                    text_encoder=text_encoder,
                                    text=c_inputs["texts"],
                                    text_null=text_null,
                                    clip_pool_features=cfg.clip_pool_features,
                                    device=device
                                )
                                print(f"\ntext_original: {c_inputs['texts']}")
                                print(f"\ntext_null: {text_null}")
                            break
                    # ===== Validate DDPMs ====

                    # ===== Sampling DDPMs ====
                    with torch.no_grad():
                        sample = ddpm_ema.ema_model.sample(
                            batch_size=cfg.batch_size_eval,
                            num_steps=cfg.diffusion_num_sampling_steps,
                            rng=torch.Generator(device=device).manual_seed(0),
                            mode=cfg.diffusion_sampling,
                            return_noise=False,

                            text_features=c_inputs["text_features"],
                            text_null_features=c_inputs["text_null_features"],
                            conditional_x_0=c_inputs["x"],
                            points=c_inputs["sampling_points"],
                            batches=c_inputs["sampling_batch"],
                            semantic=c_inputs["semantic"] if cfg.use_seg else None,
                            sampling_condition=c_inputs["sampling_depth"],
                        )
                    if(cfg.use_guidence_net and not cfg.use_control_net):
                        c_inputs["sample"], c_inputs["gl"], c_inputs["noise"] = sample
                    elif(cfg.upsampling or cfg.downsampling):
                        c_inputs["sample"], c_inputs["sparse_dense"], c_inputs["noise"] = sample
                    else:
                        c_inputs["sample"], c_inputs["noise"] = sample
                    if(cfg.use_guidence_net and not cfg.use_control_net):
                        c_inputs["gl"], _ = common.split_channels(c_inputs["gl"], cfg.train_depth, cfg.train_reflectance)
                    c_inputs["sample"],_ = common.split_channels(c_inputs["sample"], cfg.train_depth, cfg.train_reflectance)
                    # ===== Sampling DDPMs ====

                    # ==== Saving SomeThings ====
                    print()
                    lidar_utils.sample_to_lidar(
                        c_inputs["sample"],
                        g=c_inputs["gl"],
                        xyz=c_inputs["xyz"],
                        upsampling=c_inputs["sparse_dense"] if cfg.upsampling else None,
                        downsampling=c_inputs["sparse_dense"] if cfg.downsampling else None,
                        text=c_inputs["texts"],
                        semantic=c_inputs["semantic_org"] if cfg.use_seg else None,

                        num_step=global_step,
                        num_sample=cfg.diffusion_num_sampling_steps,
                        dataset=cfg.dataset,
                        # noise=noise
                    )
                    # ==== Saving SomeThings ====

                # ==== Saving Checkpoint ====
                if global_step % cfg.save_model_steps == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    ckpt = save_dir / f"diffusion_{global_step:010d}.pth"
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        ckpt,
                    )
                    print(f"Saved checkpoint : {ckpt}")
                # ==== Saving Checkpoint ====

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.num_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    '''
        ---- Setting require_grad ----
        total trainable params: 57,420,492
        trainable params: 16,432,192
        frozen params: 40,988,300
        ---- Setting require_grad ----
    '''

    # /root/.cache/huggingface/accelerate
    # /root/.cache/huggingface/accelerate/default_config.yaml
    # /ihoment/youjie10/.cache/huggingface/accelerate/default_config.yaml
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, dest="cfg")
    cfg: TrainingConfig = parser.parse_args().cfg
    train(cfg)
