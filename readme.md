https://github.com/mikeqzy/3dgs-avatar-release

# This repository has been tested on the following platform:
Python 3.7.13, PyTorch 1.12.1 with CUDA 11.6 and cuDNN 8.3.2, Ubuntu 22.04/CentOS 7.9.2009/Ubuntu 20.04

# install
conda env create -f environment.yml
conda activate 3dgs-avatar
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# train
python train.py dataset=DATASET_NAME
python train.py dataset=BUPT3

# evaluation
python render.py mode=test dataset.test_mode=view dataset=DATASET_NAME

# Visualize
python visualize.py dataset.predict_seq=0 dataset=DATASET_NAME
dataset.predict_seq为 0,1,2,3 来指定，其中dataset.predict_seq=3对应于规范渲染
dataset.predict_seq=0  filename = 'gBR_sBM_cAll_d04_mBR1_ch05_view1'
dataset.predict_seq=1  filename = 'gBR_sBM_cAll_d04_mBR1_ch06_view1'
dataset.predict_seq=2  filename = 'MPI_Limits-03099-op8_poses_view1'
dataset.predict_seq=3  filename = 'canonical_pose_view1'