# Evaluation Toolbox for LiDAR Generation

1. Please download the checkpoints (eval folder) from [QWTforHuggingFace/T2LDM](https://huggingface.co/QWTforHuggingFace/T2LDM/tree/main). <br/>
2. Put the eval/pretrained folder into the eval folder. <br/>
3. To advoid polluting the original runing environment, please create a new environment cloned by the original runing environment:
```
  1. conda create -n t2ldm_eval --clone t2ldm
  2. pip install pytorch-lightning==1.9.5 easydict==1.13
  3. Install torchsparse:
    1. sudo apt-get install libsparsehash-dev
    2. Download https://github.com/mit-han-lab/torchsparse/tree/v1.4.0
    3. zip torchsparse-1.4.0.zip, cd torchsparse-1.4.0
    4. python setup.py install
```

The original downloading from [LiDM](https://github.com/hancyran/LiDAR-Diffusion/blob/main/lidm/eval/README.md).
