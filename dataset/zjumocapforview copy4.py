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
import trimesh
from models.pose_correction.lbs import batch_rigid_transform

class ZJUMoCapDatasetforview(Dataset):
    def __init__(self, cfg, split='view'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        
        self.root_dir = cfg.root_dir
        self.refine = cfg.refine
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.subject = cfg.subject
        self.train_frames = cfg.train_frames
        self.train_cams = cfg.train_views
        self.val_frames = cfg.val_frames
        self.val_cams = cfg.val_views
        self.white_bg = cfg.white_background
        self.H, self.W = cfg.origin_hw # hardcoded original size
        self.h, self.w = cfg.img_hw
        self.smpl_type = 'smplx'

        self.faces = np.load('body_models/miscx/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/miscx/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/miscx/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/miscx/J_regressors.npz'))  
        self.v_template = np.load('body_models/miscx/v_templates.npz')['neutral']
        self.shapedirs = np.load('body_models/miscx/shapedirs_all.npz')['neutral']
        self.kintree_table = np.load('body_models/miscx/kintree_table.npy')

        self.v_template = torch.tensor(self.v_template, dtype=torch.float32, device=self.device)
        self.shapedirs = torch.tensor(self.shapedirs, dtype=torch.float32, device=self.device)
        self.J_regressor_ = torch.tensor(self.J_regressor['neutral'], dtype=torch.float32, device=self.device)
        self.parents = torch.tensor(self.kintree_table[0], dtype=torch.long, device=self.device)

        # hardcoded
        train_betas = [[-0.7631000280380249, -0.12389999628067017, 0.04749999940395355, -0.2054000049829483, -0.15369999408721924, 
                        -0.13050000369548798, -0.007300000172108412, 0.0860000029206276, -0.14980000257492065, 0.14499999582767487]]
        self.train_trans = torch.tensor([-0.002704070881009102, 0.8175977468490601, 2.692727565765381], dtype=torch.float32, device=self.device)
        self.train_root_orient = torch.tensor([3.062368154525757, 0.011700264178216457, -0.07195576280355453], dtype=torch.float32, device=self.device)

        betas = torch.tensor(train_betas, dtype=torch.float32, device=self.device)

        v_shaped = self.v_template + torch.einsum('bl,mkl->bmk', [betas, self.shapedirs])
        self.J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor_])



        if split == 'predict'or split == 'view':
            cam_names = self.cfg.predict_views
            frames = self.cfg.predict_frames
        else:
            raise ValueError

        with open(os.path.join(self.root_dir, self.subject, 'cam_params.json'), 'r') as f:
            self.cameras = json.load(f)
        if len(cam_names) == 0:
            cam_names = self.cameras['all_cam_names']



        start_frame, end_frame, sampling_rate = frames

        subject_dir = os.path.join(self.root_dir, self.subject)
        if split == 'predict'or split == 'view':
            predict_seqs = ['seq1',
                            'seq2',
                            'seq3',
                            'canonical_pose_view1',]
            predict_seq = self.cfg.get('predict_seq', 0)
            predict_seq = predict_seqs[predict_seq]
            model_files = sorted(glob.glob(os.path.join(subject_dir, predict_seq, '*.npz')))
            self.model_files = model_files
            frames = list(reversed(range(-len(model_files), 0)))
            if end_frame == 0:
                end_frame = len(model_files)
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            model_files = model_files[frame_slice]
            frames = frames[frame_slice]


        model_dict = np.load(model_files[1])
        self.smplth = model_dict['trans'].astype(np.float32)

        self.zero_trans = torch.from_numpy(self.smplth).to(torch.float32).to(self.device)
        self.zero_root_orient =  torch.tensor(model_dict['root_orient'], dtype=torch.float32, device=self.device)



        self.data = []

        for cam_idx, cam_name in enumerate(cam_names):
            cam_dir = os.path.join(subject_dir, cam_name)

            for d_idx, f_idx in enumerate(frames):
                model_file = model_files[d_idx]
                # get dummy gt...
                # img_file = glob.glob(os.path.join(cam_dir, '*.jpg'))[0]
                img_file = os.path.join(subject_dir, '1', '000000.jpg')
                # mask_file = glob.glob(os.path.join(cam_dir, '*.png'))[0]
                mask_file = os.path.join(subject_dir, '1', '000000.png')

                self.data.append({
                    'cam_idx': cam_idx,
                    'cam_name': cam_name,
                    'data_idx': d_idx,
                    'frame_idx': f_idx,
                    'img_file': img_file,
                    'mask_file': mask_file,
                    'model_file': model_file,
                })


        self.frames = frames
        self.model_files_list = model_files

        self.metadata = self.get_cano_smpl_path()


    def get_cano_smpl_path(self):
        cano_path = os.path.join(self.root_dir, self.subject, 'cano_data.npy')
        if os.path.exists(cano_path):
            cano_data = np.load(cano_path, allow_pickle=True).item()
            return cano_data
        else:
            raise FileNotFoundError(f"Cano data file not found: {cano_path}")


    def __len__(self):
        return len(self.data)
    

    def getitem(self, idx, data_dict=None):

        start = time.time()
        device = self.device

        if data_dict is None:
            data_dict = self.data[idx]
        cam_name = data_dict['cam_name']
        frame_idx = data_dict['frame_idx']
        model_file = data_dict['model_file']

        model_dict = np.load(model_file, mmap_mode='r')  
        

        smpl_params = {
            'trans': model_dict['trans'],
            'root_orient': model_dict['root_orient'],
            'pose_body': model_dict['pose_body'],
            'pose_hand': model_dict['pose_hand'],
            'betas': model_dict['betas']
        }

        smpl_tensors = {
            k: torch.from_numpy(v).pin_memory().to(device, non_blocking=True)
            for k, v in smpl_params.items()
        }
        root_orient = smpl_tensors['root_orient']
        root_orient = align_root_orient(root_orient, self.train_root_orient, self.zero_root_orient)

        
        pose = torch.cat([root_orient.squeeze(0), 
                        smpl_tensors['pose_body'], 
                        smpl_tensors['pose_hand']], dim=-1).reshape(-1, 3)

        trans = smpl_tensors['trans'] - self.zero_trans + self.train_trans

        # === 读取相机参数
        K = np.array(self.cameras[cam_name]['K'], dtype=np.float32).copy()
        R = np.array(self.cameras[cam_name]['R'], np.float32)
        T = np.array(self.cameras[cam_name]['T'], np.float32)
        T = T[:, 0]
        
        # === 焦距计算
        focal_length_x = K[0, 0].item()
        focal_length_y = K[1, 1].item()
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        pose_mat_full = axis_angle_to_matrix(pose)  # (24, 3, 3)
        pose_mat_full_ = pose_mat_full.float().to(device).unsqueeze(0)

        # === SMPL LBS 运算

        bone_transforms_02v = torch.tensor(self.metadata['bone_transforms_02v'], dtype=torch.float32, device=device)


        _, bone_transforms, _ = batch_rigid_transform(pose_mat_full_, self.J, self.parents, dtype=torch.float32)

        # === Canonical joint normalization
        minimal_shape = torch.tensor(self.metadata['minimal_shape'], dtype=torch.float32, device=device)
        center = minimal_shape.mean(0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        Jtr = torch.tensor(self.metadata['Jtr'], dtype=torch.float32, device=device)
        Jtr_norm = (Jtr - center - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm = (Jtr_norm - 0.5) * 2.0
        bone_transforms = bone_transforms.reshape(-1, 4, 4)

        bone_transforms = bone_transforms @ torch.linalg.inv(bone_transforms_02v)

        bone_transforms[:, :3, 3] += trans


        pose_mat = pose_mat_full[1:]  # 23 x 3 x 3
        pose_rot = torch.cat([torch.eye(3, device=device).unsqueeze(0), pose_mat], dim=0).reshape(-1, 9).unsqueeze(0)
        end = time.time()
        print("getitem time:", end - start)
    
        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image=None, mask=None,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=device,
            rots=pose_rot.float(),
            Jtrs=Jtr_norm.float().unsqueeze(0),
            bone_transforms=bone_transforms,
            smplth=self.train_trans.cpu().numpy(),
            height=self.h, weight=self.w,
            Height=self.H, Weight=self.W,
        )

    def __getitem__(self, idx):

        return self.getitem(idx)

    def readPointCloud(self,):

        ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')

        pcd = fetchPly(ply_path)
        
        return pcd

