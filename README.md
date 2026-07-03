# T2LDM++ (The extension version of T2LDM)
This repo is the official project repository of the paper **_T2LDM++: A Self-Conditioned Representation Guided Diffusion Model for Realistic Text-to-LiDAR Scene Generation_**. 
 -  Compared with T2LDM, **_the deeper theoretical insights_**, **_the more efficient framework design_**, and **_the more comprehensive experimental extensions_** are provided, please refer to [ [T2LDM++ paper](https://arxiv.org/pdf/2606.30147) ].
 -  Please download something from [HuggingFace](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main). <br/>
## Tasks (Please check **_the examples folder_** or **_the generate_mgpus.py file_**.):
- Unconditional Generation (KITTI-360, SemanticKITTI, nuScenes)
- Unconditional Partial Generation (nuScenes)
- Text-to-LiDAR Generation (SemanticKITTI, nuScenes)
- Zero-shot Text-to-LiDAR Generation from (semantic, Box)-to-LiDAR Generation (SemanticKITTI, nuScenes)
- Semantic-to-LiDAR Generation (SemanticKITTI, nuScenes)
- 3D Box-to-LiDAR Generation (nuScenes)
- Sparse-to-Dense Generation (nuScenes)
- Dense-to-Sparse Generation (nuScenes)
- BEV-to-LiDAR Generation (nuScenes) (BEV is the binary image, only including 0 and 1)
- Camera-to-(Partial Scene) LiDAR (nuScenes) <br/><br/>
  We provide many demos in **_the [examples](https://github.com/QWTforGithub/T2LDM_v2/tree/main/examples) folder_** (the conditional files is in **_the [examples/example_files](https://github.com/QWTforGithub/T2LDM_v2/tree/main/examples/example_files) folder_**):
  ```
    cd examples
    cd {task}_{dataset}
    python single_generate_{task}_{dataset}.py
  ```

## Overview
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)

## Installation

### Requirements
The following environment is recommended for running **_T2LDM++_** (4 NVIDIA 3090 GPUs or 8 NVIDIA 4090 GPUs):<br/>
**_If you only want to generate some LiDAR results (64 Steps or 1024 Steps, the GPU memory < 1G on BS=1), the Single 3090(24G)/3060(12G) GPU is enough!_**
- Ubuntu: 18.04 and above
- gcc/g++: 11.4 and above
- CUDA: 12.1
- PyTorch: 2.1.0
- python: 3.10

### Environment

#### Using environments.yaml (based on conda command)
```
  cd envs
  conda env create -f environment.yaml

  # If you want to conduct sparse-to-dense/dense-to-sparse experiments.
  cd ../pointops
  python setup.py install
```

#### Using requirements.txt (based on pip command)
```
  conda create -n t2ldm python=3.10 -y
  conda activate t2ldm

  cd envs

  pip install -r requirements.txt

  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  pip install ema-pytorch==0.4.8 kornia==0.7.0 accelerate==0.22.0

  # If you want to conduct sparse-to-dense/dense-to-sparse experiments.
  cd ../pointops
  python setup.py install
```

## Data Preparation

### nuScenes
  1. Download the official [nuScenes](https://www.nuscenes.org/nuscenes#download) (or [Baidu Disk](https://pan.baidu.com/s/1Rsbi-Q_2EUm05lwQgn8T3Q?pwd=1111)(code:1111)) dataset (with Lidar Segmentation).
  2. Put [ [nuscenes.pkl, nuscenes_camera.pkl, nuscenes_description_plus_plus.pkl](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main/nuScenes) ] into the **_../nuScenes/v1.0-trainval_** folder.
  3. Put  [ [SEMANTIC.ZIP](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main/nuScenes) ] into  the **_../nuScenes/v1.0-trainval/samples_** folder (unzip the SEMANTIC.ZIP file).
  4. The final **_../nuScenes/v1.0-trainval_** folder as follows:
  ```
  ../nuscenes/v1.0-trainval
  │── samples
  │── sweeps
  │── lidarseg
  ...
  │── v1.0-trainval 
  │── v1.0-test
  │── nuscenes.pkll
  │── nuscenes_camera.pkl
  │── nuscenes_description_plus_plus.pkl
  ```
  5. If you want to conduct the Sparse-to-Dense and Dense-to-Sparse generation, please produce the samples using the downsampling_LIDAR function in [ [descriptor.py](https://github.com/QWTforGithub/T2LDM_v2/blob/main/data/nuScenes/descriptor.py) ]:
  ```
  Please produce 0.125/0.25/0.5/1.0/2.0 samples. For example, you want to generate 0.125 samples:
  downsampling_LIDAR(
        root_path = "nuScenes/v1.0-trainval/samples/LIDAR_TOP",
        dest_path = "nuScenes/v1.0-trainval/samples/LIDAR_TOP_DOWNSAMPLING",
        up_rate=0.125
  )
  # the final nuScenes/v1.0-trainval/samples folder as follows:
  │── CAM_BACK
  │── CAM_BACK_LEFT
  │── CAM_BACK_RIGHT
  ...
  │── LIDAR_TOP_DOWNSAMPLING0.125
  │── LIDAR_TOP_DOWNSAMPLING0.25
  │── LIDAR_TOP_DOWNSAMPLING0.5
  │── LIDAR_TOP_DOWNSAMPLING1.0
  │── LIDAR_TOP_DOWNSAMPLING2.0
  │── SEMANTIC
  ```

### SemanticKITTI
  1. Dowload the official [SemanticKITTI (https://semantic-kitti.org/dataset.html).
  2. Put [ [kitti_semantic.pkl, semantic_kitti_description.pkl](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main/SemanticKITTI) ] into **_the ../SemanticKITTI/dataset/sequences folder_**.
  3. The final **_../SemanticKITTI/sequences_** folder as follows:
```
  ../SemanticKITTI/dataset/sequences
  │── 00
  │── 01
  │── 02
  ...
  │── 20
  │── 21
  │── kitti_semantic.pkl
  │── semantic_kitti_description.pkl
```

### KITTI360
  Dowload the official [KITTI360 (Raw Velodyne Scans (119G))](https://www.cvlibs.net/datasets/kitti-360/download.php) and organize the download files as follows:
```
  ../KITTI360/data_3d_raw
  │── 2013_05_28_drive_0000_sync
  │── 2013_05_28_drive_0002_sync
  │── 2013_05_28_drive_0003_sync
  ...
  │── 2013_05_28_drive_0009_sync
  │── 2013_05_28_drive_0010_sync
```

## Model Zoo
We build a Huggingface project [QWTforHuggingFace/T2LDMv2](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main). Please download [checkpoints](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main/checkpoints) from Huggingface.<br/>

## Quick Start

### Accelerate Configuration
Before the training and sampling, it must deploys the accelerate.
```
  conda activate t2ldm
  accelerate config
  # please finsh the accelerate configuration according to the tips.
```

### Domes
Please check **_the [examples](https://github.com/QWTforGithub/T2LDM_v2/tree/main/examples) folder_**.

### Training
Please check **_the train_{task}_{dataset}_gn.py file_**.

### Sampling
Please check **_the [generate_mgpus.py](https://github.com/QWTforGithub/T2LDM_v2/blob/main/generate_mgpus.py) file_**.
