import dataclasses
import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclasses.dataclass
class TrainingConfig:

    # ---- dataset setting ----
    dataset: Literal["kitti_semantic", "kitti_360", "nuScenes"] = "nuScenes"
    data_root: str = "/root/dataset/nuScenes"
    aug: tuple[int, int] = ("-", "-") # ("rotation", "flip")
    resolution: tuple[int, int] = (32, 192)
    depth_range: tuple[int, int] = (0.01, 50.0) # (1.45, 80.0) # (0.01, 50.0)
    fov: tuple[int, int] = (3, -25)
    batch_size_train: int = 128#64 # 2 # 16
    batch_size_eval: int = 4
    num_workers: int = 32#16 # 2 # 16

    text_keys: str = "text" # "text_l0 text_l1 text_2"
    pkl: str = "nuscenes_camera.pkl" # "nuscenes_description_plus_plus.pkl"
    version: str = "v1.0-trainval"
    semantic_class_num: float = 17.0
    use_3dbox: bool = False
    use_camera: bool = True
    use_bev: bool = False
    use_seg: bool = False
    use_zeroshot_text: bool = False
    only_class: int = -1
    # ---- dataset setting ----

    # ---- model setting ----
    model_name: str = "circular_unet"

    in_channel: int = 2
    out_channel: int = 2
    control_channel:  int = 128 #1

    n_base_channel: int = 64
    n_channels = [1, 2, 4, 4]
    n_strides = [[1, 2], [2, 2], [2, 2]]
    n_attn_types = ["linear", "linear", "linear", "vanilla"] # ["nn", "nn", "nn", "nn"] # ["linear", "linear", "linear", "vanilla"]
    n_norm_types = ["gn", "gn", "gn", "gn"] # ["ln", "ln", "ln", "ln"] # ["gn", "gn", "gn", "gn"]
    n_use_norm = [True, True, True, True]
    n_use_res_connection = [True, True, True, True]
    n_heads = [2, 4, 8, 8]

    n_midd_attn_type: str = 'vanilla' # 'nn' # 'vanilla'
    n_midd_norm_type: str = 'gn' # 'ln' # 'gn'
    n_midd_use_norm: bool = True
    n_midd_use_res_connection: bool = True
    n_midd_heads: int = 8
    n_midd_channels = [4, 8, 4]

    g_base_channel: int = 64
    g_channels = [1, 2, 4, 4]
    g_strides = [[1, 2], [2, 2], [2, 2]]
    g_attn_types = ["linear", "linear", "linear", "vanilla"]
    g_norm_types = ["gn", "gn", "gn", "gn"]
    g_use_norm = [True, True, True, True]
    g_use_res_connection = [True, True, True, True]
    g_heads = [2, 4, 8, 8]

    g_midd_attn_type: str = 'vanilla'
    g_midd_norm_type: str = 'gn'
    g_midd_use_norm: bool = True
    g_midd_use_res_connection: int = True
    g_midd_heads: int = 8
    g_midd_channels = [4, 8, 4]

    dropout: float = 0.0
    resamp_with_conv: bool = True
    skip_connection_scale: str = "sqrt(2)"  # equal, sqrt(2), scalelong

    text_channels: int = 768
    t_channels: int = 384

    use_pe: bool = False
    use_rope: bool = True
    use_dpe_n: bool = True
    use_dpe_g: bool = False
    use_zero_weight: bool = True
    use_circularconv_shortcut: bool = False # using Conv2D or CircularConv2D in ResBlock

    use_guidence_net: bool = True
    use_control_net: bool = False
    use_image_encoder: bool = False
    freeze_guidence_net: bool = False
    use_multistage_feats: bool = False
    attention_gate: str = False # "head" "element" False

    use_text: bool = False # Text-Guidance
    clip_mode: str = "ViT-L/14" # "ViT-B/32", None
    clip_pool_features: bool = False # True -> [B,768/512], False -> [B,77,768/512]

    T5_mode: str = "base" # small, base, large, None

    print_channels: bool = False
    print_info: bool = False

    upsampling: bool = False
    up_rate: float = 4
    base_up_rate: float = 0.25
    downsampling: bool = False
    down_rate: float = 0.5
    base_down_rate: float = 1.0
    interpolate_rate: float = 0.5
    # ---- model setting ----

    # ---- training setting ----
    image_format: str = "log_depth"
    lidar_projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    train_mask: bool = True # True
    num_steps: int = 400_000 # 8 GPUs ==> 400_000, 4 GPUs ==> 800_000
    save_sample_steps: int =  5_000#10_000# 5_000 # 10_000 # 1
    save_model_steps: int =  5_000#10_000 # 1
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4 # 1e-4， 1e-5(效果一般)
    lr_warmup_steps: int = 2_000 # 16_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999 # 0.99
    adam_weight_decay: float = 0.01 # 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.9997 # 0.995
    ema_update_every: int = 1 # 10 # 20
    output_dir: str = "logs/diffusion"
    seed: int = 0
    mixed_precision: str = "no" # "fp16", "no"
    dynamo_backend: str = None # "inductor", "no", None
    filer_keys = None #[".n_",".temb.",".g_",] # ["n_","temb","g_",] # None
    checkpoint_dir: str = None
    pretrained_checkpoint_dir: str = None
    # ---- training setting ----

    # ---- diffusion setting ----
    criterion: str = "huber"#"l2"
    diffusion_num_training_steps: int = 1024 #2048
    diffusion_num_sampling_steps: int = 1024 #2048
    diffusion_objective: Literal["eps", "v", "x0"] = "v"
    diffusion_prediction: Literal["eps", "v", "x0"] = "v"
    diffusion_beta_schedule: str = "cosine"
    diffusion_timesteps_type: Literal["continuous", "discrete"] = "continuous"
    diffusion_sampling: str = "ddpm" # "ddpm", "ddim"
    diffusion_min_snr_loss_weight: bool = True
    diffusion_min_snr_gamma: float = 5.0
    diffusion_clip_sample: bool = True
    diffusion_clip_sample_range: float = 1.0

    diffusion_guidence_step_interval: int = 100_000
    diffusion_guidence_weights = [0.001, 0.01, 0.01, 0.01]

    diffusion_classifier_free_scale: float = None  # 4.0
    diffusion_classifier_dropout: float = 0.0 # 0.1
    # ---- diffusion setting ----
