https://github.com/mikeqzy/3dgs-avatar-release

# This repository has been tested on the following platform:
Python 3.7.13, PyTorch 1.12.1 with CUDA 11.6 and cuDNN 8.3.2, Ubuntu 22.04/CentOS 7.9.2009/Ubuntu 20.04

## Installation

To clone the repo, run either:
```
git clone --recursive https://github.com/yahui-lee/x_avatar.git
```
or
```
git clone https://github.com/yahui-lee/x_avatar.git
cd 3dgs-avatar-release
git submodule update --init --recursive
```

Next, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `3dgs-avatar` using
```
conda env create -f environment.yml
conda activate 3dgs-avatar
# install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
### SMPL-X Setup
Download `SMPLX v1.1` from [SMPL website](https://smpl-x.is.tue.mpg.de/download.php). Remove the chumpy objects in these .pkl models using [this code](https://github.com/vchoutas/smplx/tree/master/tools) under a Python 2 environment (you can create such an environment with conda). Finally, rename the newly generated .pkl files and copy them to subdirectories under `./body_models/smplx/`. Eventually, the `./body_models` folder should have the following structure:
```
body_models
 └-- smplx
    ├-- male
    |   └-- model.pkl
    ├-- female
    |   └-- model.pkl
    └-- neutral
        └-- model.pkl
```

Then, run the following script to extract necessary SMPLX parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be saved into `./body_models/miscx/`.

# Dataset preparation
Due to license issues, we cannot publicly distribute our preprocessed ZJU-MoCap data. 
Please follow the instructions of [ARAH](https://github.com/taconite/arah-release) to download and preprocess the datasets.


# train
```
python train.py dataset=DATASET_NAME
python train.py dataset=BUPT3
```

# Evaluation
```
python render.py mode=test dataset.test_mode=view dataset=DATASET_NAME
```
# Test on out-of-distribution poses
To animate the subject under out-of-distribution poses, run
```shell
python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

We provide four preprocessed sequences for each subject of ZJU-MoCap, 
which can be specified by setting `dataset.predict_seq` to 0,1,2,3, 
where `dataset.predict_seq=3` corresponds to the canonical rendering.

Currently, the code only supports animating ZJU-MoCap models for out-of-distribution models.

# Visualize
```
python visualize.py dataset.predict_seq=0 dataset=DATASET_NAME
dataset.predict_seq为 0,1,2,3 来指定，其中dataset.predict_seq=3对应于规范渲染
dataset.predict_seq=0  filename = 'gBR_sBM_cAll_d04_mBR1_ch05_view1'
dataset.predict_seq=1  filename = 'gBR_sBM_cAll_d04_mBR1_ch06_view1'
dataset.predict_seq=2  filename = 'MPI_Limits-03099-op8_poses_view1'
dataset.predict_seq=3  filename = 'canonical_pose_view1'
```
注意！！
1. 本项目只支持smplx，且参数均设置为训练和驱动阿里数据集，训练其他数据需稍微修改超参数。
2. 项目中的Visualize代码不完整，因为查看器的完整代码为单独一个项目，在之后会添加进去。
3. 本项目用到的所有数据集均处理为Zju-mocap格式，所以即使不用zjumocap数据集，都要用到arah代码里的方法处理。

# ALi Data
关于阿里，数据集，提供dataprocess_ali.py，方便处理此数据，具体输入（处理原始数据格式，方便后续处理）和输出（经处理，可以被x_avatar读取，训练)参考ali.json和ali.npz
注：
1. 上述为smplx数据的处理，其他数据（如图片，掩膜，相机参数）的摆放如下，相机参数格式参考cam_params.json（只需要修改K,D,R,T)：

```
/data/ZJUMoCap/ali/
├── cam_params.json         # 相机参数文件
├── cano_smpl.ply           # SMPL模型文件
├── 1/                      # 相机1的图片文件夹
│   ├── 000000.jpg
│   ├── 000000.png
│   ├── 000001.jpg
│   ├── 000001.png
│   └── ...
├── 2/                      # 相机2的图片文件夹
│   └── ...
├── 3/
│   └── ...
├── ...                     # 依次到59
├── 59/
│   └── ...
└── models/                 # smplx参数
│   └── ...000000.npz
```
2. 阿里数据集的smplx全局旋转和平移的前向蒙皮过程和官方不一样，需要进行修正，这里给出修正代码（Talkbody/fix_error_ali.py），根据需求自行修改
```
origindata -> origindata(fix_error_ali.py) -> ali.json -> ali.npz
```
