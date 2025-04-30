import os
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB, align_root_orient
from scene.cameras import Camera
from utils.camera_utils import freeview_camera
from pytorch3d.transforms import axis_angle_to_matrix
import time
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from models.pose_correction.lbs import batch_rigid_transform

class ZJUMoCapDatasetforview(Dataset):
    def __init__(self, cfg, split='view'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.root_dir = cfg.root_dir
        self.subject = cfg.subject
        self.H, self.W = cfg.origin_hw
        

        # 加载 SMPL 模型参数
        self.v_template = torch.tensor(np.load('body_models/miscx/v_templates.npz')['neutral'], dtype=torch.float32, device=self.device)
        self.shapedirs = torch.tensor(np.load('body_models/miscx/shapedirs_all.npz')['neutral'], dtype=torch.float32, device=self.device)
        self.J_regressor = torch.tensor(np.load('body_models/miscx/J_regressors.npz')['neutral'], dtype=torch.float32, device=self.device)
        self.parents = torch.tensor(np.load('body_models/miscx/kintree_table.npy')[0], dtype=torch.long, device=self.device)

        # hardcoded
        if self.cfg.subject == 'LYH3':
            train_betas = torch.tensor([[-0.7631000280380249, -0.12389999628067017, 0.04749999940395355, -0.2054000049829483, -0.15369999408721924, 
                            -0.13050000369548798, -0.007300000172108412, 0.0860000029206276, -0.14980000257492065, 0.14499999582767487]], dtype=torch.float32, device=self.device)
            self.train_trans = torch.tensor([-0.002704070881009102, 0.8175977468490601, 2.692727565765381], dtype=torch.float32, device='cpu')
            self.train_root_orient = torch.tensor([3.062368154525757, 0.011700264178216457, -0.07195576280355453], dtype=torch.float32, device='cpu')
        elif self.cfg.subject == 'LYH4':
            train_betas = torch.tensor([[0.33576292,-0.36653692,0.44230965,-0.5486812,-0.07879381,-0.10152555,-0.0096653,-0.1306343,-0.4110131,-0.036776]], dtype=torch.float32, device=self.device)
            self.train_trans = torch.tensor([-0.0999622,0.3905677,3.4824572], dtype=torch.float32, device='cpu')
            #self.train_root_orient = torch.tensor([1, 0, 0], dtype=torch.float32, device='cpu')
            self.train_root_orient = torch.tensor([-3.08215,0.00382915,0.03496685], dtype=torch.float32, device='cpu')     
        self.train_root_orient = axis_angle_to_matrix(self.train_root_orient.unsqueeze(0))
        # SMPL Joint 推导
        v_shaped = self.v_template + torch.einsum('bl,mkl->bmk', train_betas, self.shapedirs)
        self.J = torch.einsum('bik,ji->bjk', v_shaped, self.J_regressor)

        # 相机参数
        cam_path = os.path.join(self.root_dir, self.subject, 'cam_params.json')
        with open(cam_path, 'r') as f:
            self.cameras = json.load(f)
        
        self.R = np.array(self.cameras['1']['R'], dtype=np.float32)
        self.T = np.array(self.cameras['1']['T'], dtype=np.float32)[:, 0]


        # Canonical mesh 数据
        self.metadata = self.get_cano_metadata()


        # === Canonical joint normalization
        minimal_shape = torch.tensor(self.metadata['minimal_shape'], dtype=torch.float32)
        center = minimal_shape.mean(0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        Jtr = torch.tensor(self.metadata['Jtr'], dtype=torch.float32)
        Jtr_norm = (Jtr - center - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm = (Jtr_norm - 0.5) * 2.0
        self.Jtr_norm = Jtr_norm.float().unsqueeze(0).to(self.device)

        self.zero_trans = None
        self.zero_root_orient = None
        self.dummy = 0

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx, data_dict):
        assert data_dict is not None, "Must provide `data_dict` containing 'cam_name', 'frame_idx', 'smpl_params'."
        self.h, self.w = self.cfg.img_hw
        device = self.device
        cam_name = '1'
        frame_idx = data_dict['frame_idx']
        smpl_input = data_dict['smpl_params'] 

        # 转为 CPU tensor
        smpl_tensors = {}
        for k in ['trans', 'root_orient', 'pose_body', 'pose_hand', 'betas']:
            v = smpl_input[k]
            if isinstance(v, np.ndarray):
                smpl_tensors[k] = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                smpl_tensors[k] = v.cpu()
            else:
                raise TypeError(f"Unsupported type for {k}: {type(v)}")

        # 初始化
        if frame_idx == 0 and not self.dummy:
            self.dummy = 1
            self.zero_trans = smpl_tensors['trans']
            self.zero_root_orient = axis_angle_to_matrix(smpl_tensors['root_orient'].unsqueeze(0))


        root_orient = align_root_orient(smpl_tensors['root_orient'], self.train_root_orient, self.zero_root_orient)
        pose = torch.cat([root_orient.squeeze(0), smpl_tensors['pose_body'], smpl_tensors['pose_hand']], dim=-1).reshape(-1, 3)

        trans = self.train_trans
        K = np.array(self.cameras['1']['K'], dtype=np.float32)
        self.FoVx = focal2fov(K[0, 0].item(), self.w)
        self.FoVy = focal2fov(K[1, 1].item(), self.h)

        pose_mat = axis_angle_to_matrix(pose).float().unsqueeze(0)
        _, bone_transforms, _ = batch_rigid_transform(pose_mat, self.J.cpu(), self.parents.cpu())
        

        bone_transforms = bone_transforms.reshape(-1, 4, 4)
        bone_transforms = bone_transforms @ torch.linalg.inv(torch.tensor(self.metadata['bone_transforms_02v'], dtype=torch.float32))
        bone_transforms[:, :3, 3] += trans

        # 仅将 rots 和 Jtrs 放回 GPU
        pose_rot = torch.cat([torch.eye(3).unsqueeze(0), pose_mat.squeeze(0)[1:]], dim=0).reshape(-1, 9).unsqueeze(0).to(device)
        Jtr_norm = self.Jtr_norm

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=self.R, T=self.T,
            FoVx=self.FoVx, FoVy=self.FoVy,
            image=None, mask=None,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{abs(frame_idx):06d}",
            data_device=device,
            rots=pose_rot,
            Jtrs=Jtr_norm,
            bone_transforms=bone_transforms,
            smplth=self.train_trans.numpy(),
            height=self.h, weight=self.w,
            Height=self.H, Weight=self.W,
        )


    def get_cano_metadata(self):
        cano_path = os.path.join(self.root_dir, self.subject, 'cano_data.npy')
        if os.path.exists(cano_path):
            return np.load(cano_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"Missing canonical data file: {cano_path}")

    def readPointCloud(self):
        ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
        return fetchPly(ply_path)
