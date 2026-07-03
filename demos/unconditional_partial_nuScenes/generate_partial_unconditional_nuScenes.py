import argparse
import json
import os
import dataclasses
import torch
import datetime
import utils.render
import shutil
import inspect
from pathlib import Path
from accelerate import Accelerator
from utils import common
import torch.distributed as dist
from simple_parsing import ArgumentParser
import time

from data.conditional_x0.conditionalx0 import ConditionalX0
from torch.utils.data import DataLoader
from models.CLIP.clip import clip
from models.T5.T5 import t5

import sys

sys.path.append(".")

from models.T2LDM_plus_plus import CircularUNet
from examples.unconditional_partial_nuScenes.config_partial_unconditional_nuScenes_gn import TrainingConfig

import examples.unconditional_partial_nuScenes.loading_partial_unconditional_nuScenes as inference_mgpus


def main(args, cfg):
    # ---- Log Path ----
    task = inspect.getfile(TrainingConfig).split("/")[-1].split("_")[1]
    project_dir = args.project_dir
    now = datetime.datetime.now()
    second = (now.second // 20) * 20
    project_name = now.replace(second=second, microsecond=0).strftime("%Y%m%dT%H%M%S")
    project_name = f"{project_name}_{args.num_steps}_{args.sampling_mode}{args.sampling_steps}_{task}_{cfg.dataset}_{args.seed}"
    dest_path = os.path.join(project_dir, project_name)
    os.makedirs(dest_path, exist_ok=True)
    # ---- Log Path ----

    # ---- Accelerator Config ----
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="no",
        log_with=["tensorboard"],
        project_dir=dest_path,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )

    device = accelerator.device

    if accelerator.is_main_process:

        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )

        config_path = inspect.getfile(TrainingConfig)
        config_dest_path = os.path.join(dest_path, config_path.split("/")[-1])
        if (os.path.exists(config_path)):
            shutil.copy(config_path, config_dest_path)
            print(f"Coping config file to {config_dest_path}")

        net_path = inspect.getfile(CircularUNet)
        net_dest_path = os.path.join(dest_path, net_path.split("/")[-1])
        if (os.path.exists(net_path)):
            shutil.copy(net_path, net_dest_path)
            print(f"Coping network file to {net_dest_path}")

        inference_path = str(Path(inference_mgpus.__file__).resolve())
        inference_dest_path = os.path.join(dest_path, inference_path.split("/")[-1])
        if (os.path.exists(inference_path)):
            shutil.copy(inference_path, inference_dest_path)
            print(f"Coping inference file to {inference_dest_path}")

        generate_path = str(Path(__file__).resolve())
        generate_dest_path = os.path.join(dest_path, generate_path.split("/")[-1])
        if (os.path.exists(generate_path)):
            shutil.copy(generate_path, generate_dest_path)
            print(f"Coping generate file to {generate_dest_path}")

        train_name = config_path.split("/")[-1].replace("config_", "train_")
        training_path = f"{str(Path(__file__).parent.resolve())}/{train_name}"
        training_dest_path = os.path.join(dest_path, training_path.split("/")[-1])
        if (os.path.exists(training_path)):
            shutil.copy(training_path, training_dest_path)
            print(f"Coping train file to {training_dest_path}")

        print("\nAccelerator配置信息: ")
        print(accelerator.state)
    # ---- Accelerator Config ----

    # ---- Model Config ----
    ddpm, lidar_utils, _ = inference_mgpus.setup_model(
        ckpt_path=args.ckpt,
        device=args.device,
        load_config=args.load_config,
        project_dir=dest_path
    )

    ddpm = accelerator.prepare(ddpm)
    # ---- Model Config ----

    # ---- Random Seed ----
    seed = common.setup_seed(args.seed)
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] Random check: {torch.randint(0, 10000, (1,))}")
    # ---- Random Seed ----

    # ---- Validation Dataset ----
    inputs = common.get_inputs()
    if (cfg.use_text or cfg.use_camera or cfg.use_control_net):
        condition_guide_dataset = ConditionalX0(
            data_root=cfg.data_root,
            pkl=cfg.pkl,

            semantic_class_num=cfg.semantic_class_num,

            training=True,
            aug=cfg.aug,

            resolution=cfg.resolution,
            depth_range=cfg.depth_range,
            fov=cfg.fov,

            use_seg=cfg.use_seg,
            use_3dbox=cfg.use_3dbox,
            use_camera=cfg.use_camera,
            use_bev=cfg.use_bev,

            sampling=True if cfg.upsampling or cfg.downsampling else False,
            up_rate=cfg.base_up_rate,
            down_rate=cfg.base_down_rate,

            random_num=args.random_num,
            type=cfg.dataset,
        )

        condition_guide_dataloader = DataLoader(
            condition_guide_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=common.collate_fn
        )

        for batch in condition_guide_dataloader:
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
            break

        if (cfg.use_text):

            print(f"\ntext_all: {inputs['texts']}")

            if (not cfg.use_zeroshot_text):
                # ---- Text Encoder ----
                text_encoder = None
                if (cfg.use_text):
                    if (cfg.clip_mode is not None):
                        text_encoder = clip.load(cfg.clip_mode, device=device)
                    elif (cfg.T5_mode is not None):
                        text_encoder = t5(cfg.T5_mode).to(device)
                # ---- Text Encoder ----

                inputs['texts_null'] = [""] * len(inputs["texts"])
                print(f"\ntext_null: {inputs['texts_null']}")
                inputs["text_features"], inputs["text_null_features"] = common.get_text_features(
                    text_encoder=text_encoder,
                    text=inputs["texts"],
                    text_null=inputs['texts_null'],
                    clip_pool_features=cfg.clip_pool_features,
                    device=device
                )
    # ---- Validation Dataset ----

    # ---- Sampling ----
    start_time = time.time()
    sample = accelerator.unwrap_model(ddpm).sample(
        text_features=inputs["text_features"],
        text_null_features=inputs["text_null_features"],
        semantic=inputs["semantic"] if cfg.use_seg else None,
        points=inputs["sampling_points"],
        batches=inputs["sampling_batch"],
        sampling_condition=inputs["sampling_depth"],

        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        return_noise=False,
        mode=args.sampling_mode,
    )
    all_time = time.time() - start_time
    avg_time = all_time / 4 / args.batch_size
    print(f"all time : {all_time}, avg time : {avg_time}")

    if (cfg.upsampling or cfg.downsampling):
        inputs["sample"], inputs["sparse_dense"], inputs["noise"] = sample
    else:
        inputs["sample"], inputs["noise"] = sample
    inputs["sample"] = inputs["sample"].clamp(-1, 1)
    inputs["sample"], _ = common.split_channels(inputs["sample"], cfg.train_depth, cfg.train_reflectance)
    # ---- Sampling ----

    # ---- Saving Something ----
    process = accelerator.process_index
    lidar_utils.sample_to_lidar(
        inputs["sample"],
        g=inputs["gl"],
        xyz=inputs["xyz"],
        noise=inputs["noise"],
        upsampling=inputs["sparse_dense"] if cfg.upsampling else None,
        downsampling=inputs["sparse_dense"] if cfg.downsampling else None,
        text=inputs["texts"],
        semantic=inputs["semantic_org"] if cfg.use_seg and not cfg.use_camera and not cfg.use_bev else None,
        box=inputs["box"] if cfg.use_3dbox else None,
        camera=inputs["camera"] if cfg.use_camera and not cfg.use_bev else None,
        camera_info=inputs["camera_info"] if cfg.use_camera else None,
        bev=inputs["bev"] if cfg.use_bev else None,

        num_step=args.num_steps,
        num_sample=args.sampling_steps,
        rank=seed,
        process=process,
        dataset=cfg.dataset,
    )
    # ---- Saving Something ----


if __name__ == "__main__":

    root_path =  str(Path(__file__).resolve().parent.parent.parent)
    ckpt = "/root/models/checkpoints/nuScenes_partial_unconditional_0000400000.pth"

    seed = 81  # 81, 309, 88, 127, 744,          830, 171,18, 256, 803, 201,
    batch_size = 4  # 32  # 64
    sampling_steps = 1024  # 64 #1024
    sampling_mode = "ddpm"
    num_steps = int(ckpt.split("?")[-1].split("_")[-1].split(".")[0])
    # num_steps = 400_000
    random_num = 256
    project_dir = "examples/unconditional_partial_nuScenes"
    project_dir = f"{root_path}/{project_dir}/test_unconditional_partial_nuScenes"
    load_config = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ckpt)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--sampling_steps", type=int, default=sampling_steps)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--num_steps", type=int, default=num_steps)
    parser.add_argument("--sampling_mode", type=str, default=sampling_mode)
    parser.add_argument("--random_num", type=int, default=random_num)
    parser.add_argument("--project_dir", type=str, default=project_dir)
    parser.add_argument("--load_config", type=bool, default=load_config)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    parser_cfg = ArgumentParser()
    parser_cfg.add_arguments(TrainingConfig, dest="cfg")
    cfg: TrainingConfig = parser_cfg.parse_args().cfg

    main(args, cfg)