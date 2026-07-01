# TBK
1. Please download the organized FSHNet (the TBK folder) from [HuggingFace](https://huggingface.co/QWTforHuggingFace/T2LDMv2/tree/main/TBK). <br/>
2. zip TBK.zip
3. Install the runing environment
```
  conda create -n tbk python=3.8 -y
  conda activate tbk
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  
  cd TBK/envs
  pip install -r requirements1.txt
  pip install -r requirements2.txt
  pip install -r requirements3.txt
  pip install -r requirements4.txt
  pip install -r requirements5.txt
  pip install -r requirements6.txt
  pip install -r requirements7.txt
  pip install -r requirements8.txt
  pip install -r requirements9.txt
  
  cd ../FSHNet-TBK
  python setup.py develop
```
4. Predict the 3D Box based on the generation LiDAR scenes:
```
  cd tools
  # In the test.py file, 'generation_folder' means the generated data folder and 'box_folder' indicates the predicted 3D box folder.
  python test.py
  # This will produce the pkl file including the predicted 3D boxes.
```

5. Conduct the TBK result.
```
  Please check the TBK.py file, you will understand how to get the TBK result.
```
