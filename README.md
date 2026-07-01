# T2LDM++ (The extension version of T2LDM)
This repo is the official project repository of the paper **_T2LDM++: A Self-Conditioned Representation Guided Diffusion Model for Realistic Text-to-LiDAR Scene Generation_**. 
 -  Compared with T2LDM, **_the deeper theoretical insights_**, **_the more efficient framework design_**, and **_the more comprehensive experimental extensions_** are provided, please refer to [ [T2LDM++ paper](https://arxiv.org/pdf/2606.30147) ].
 -  Please download something from [HuggingFace](https://huggingface.co/QWTforHuggingFace/T2LDMv2). <br/>
## Tasks:
- Unconditional Generation (KITTI-360, SemanticKITTI, nuScenes)
- Unconditional Partial Generation (nuScenes)
- Text-to-LiDAR Generation (SemanticKITTI, nuScenes)
- Zero-shot Text-to-LiDAR Generation from (semantic, Box)-to-LiDAR Generation (SemanticKITTI, nuScenes)
- Semantic-to-LiDAR Generation (SemanticKITTI, nuScenes)
- 3D Box-to-LiDAR Generation (nuScenes)
- Sparse-to-Dense Generation (nuScenes)
- Dense-to-Sparse Generation (nuScenes)
- BEV-to-LiDAR Generation (nuScenes) (BEV is the binary image, only including 0 and 1)
- Camera-to-(Partial Scene) LiDAR (nuScenes)
- Please check **_the examples folder_** or **_the generate_mgpus.py file_**.

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
