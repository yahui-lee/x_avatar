import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera
from utils.camera_utils import freeview_camera
from pytorch3d.transforms import axis_angle_to_matrix
import time
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh
from models.pose_correction.lbs import blend_shapes,vertices2joints,batch_rodrigues,batch_rigid_transform
class ZJUMoCapDatasetforview(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        
        self.root_dir = cfg.root_dir
        self.refine = cfg.refine
        if self.refine:
            self.root_dir = "../../data/refined_ZJUMoCap_arah_format"
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

        if self.smpl_type == 'smpl':
            self.faces = np.load('body_models/misc/faces.npz')['faces']
            self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
            self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
            self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))
        else:
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

        if split == 'train':
            cam_names = self.train_cams
            frames = self.train_frames
        elif split == 'val':
            cam_names = self.val_cams
            frames = self.val_frames
        elif split == 'test':
            cam_names = self.cfg.test_views[self.cfg.test_mode]
            frames = self.cfg.test_frames[self.cfg.test_mode]
        elif split == 'predict'or split == 'view':
            cam_names = self.cfg.predict_views
            frames = self.cfg.predict_frames
        else:
            raise ValueError

        with open(os.path.join(self.root_dir, self.subject, 'cam_params.json'), 'r') as f:
            self.cameras = json.load(f)

        if len(cam_names) == 0:
            cam_names = self.cameras['all_cam_names']
        elif self.refine:
            cam_names = [f'{int(cam_name) - 1:02d}' for cam_name in cam_names]


        start_frame, end_frame, sampling_rate = frames

        subject_dir = os.path.join(self.root_dir, self.subject)
        if split == 'predict'or split == 'view':
            predict_seqs = ['gBR_sBM_cAll_d04_mBR1_ch05_view1',
                            'gBR_sBM_cAll_d04_mBR1_ch06_view1',
                            'MPI_Limits-03099-op8_poses_view1',
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
        else:
            if self.cfg.get('arah_opt', False):
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'opt_models/*.npz')))
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))
            self.model_files = model_files
            frames = list(range(len(model_files)))
            if end_frame == 0:
                end_frame = len(model_files)
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            model_files = model_files[frame_slice]
            frames = frames[frame_slice]

        model_dict = np.load(model_files[1])
        self.smplth = model_dict['trans'].astype(np.float32)
        # add freeview rendering
        if cfg.freeview:
            # with open(os.path.join(self.root_dir, self.subject, 'freeview_cam_params.json'), 'r') as f:
            #     self.cameras = json.load(f)
            self.cameras = freeview_camera(self.cameras[cam_names[0]], self.smplth)
            cam_names = self.cameras['all_cam_names']

        betas = torch.tensor(model_dict['betas'], dtype=torch.float32, device=self.device)

        v_shaped = self.v_template + torch.einsum('bl,mkl->bmk', [betas, self.shapedirs])
        self.J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor_])

        self.data = []
        if split == 'predict' or cfg.freeview or split == 'view':
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
        else:
            for cam_idx, cam_name in enumerate(cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)
                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[frame_slice]
                mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[frame_slice]

                for d_idx, f_idx in enumerate(frames):
                    img_file = img_files[d_idx]
                    mask_file = mask_files[d_idx]
                    model_file = model_files[d_idx]

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

        self.get_metadata()

        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]



    def get_metadata(self):
        data_paths = self.model_files
        data_path = data_paths[0]

        #cano_data = self.get_cano_smpl_verts(data_path)
        cano_data = self.get_cano_smpl_path()
        if self.split != 'train':
            self.metadata = cano_data
            return

        start, end, step = self.train_frames
        frames = list(range(len(data_paths)))
        if end == 0:
            end = len(frames)
        frame_slice = slice(start, end, step)
        frames = frames[frame_slice]

        frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        }
        self.metadata.update(cano_data)
        if self.cfg.train_smpl:
            self.metadata.update(self.get_smpl_data())

    def get_cano_smpl_path(self):
        cano_path = os.path.join(self.root_dir, self.subject, 'cano_data.npy')
        if os.path.exists(cano_path):
            cano_data = np.load(cano_path, allow_pickle=True).item()
            return cano_data

    def get_cano_smpl_verts(self, data_path):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''
        # compute scale from SMPL body
        model_dict = np.load(data_path)
        gender = 'neutral'

        # 3D models and points
        minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            'gender': gender,
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v,
            'cano_mesh': cano_mesh,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def get_smpl_data(self):
        # load all smpl fitting of the training sequence
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_files_list)):
            model_dict = np.load(model_file)

            if idx == 0:
                smpl_data['betas'] = model_dict['betas'].astype(np.float32)

            smpl_data['frames'].append(frame)
            smpl_data['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            smpl_data['pose_body'].append(model_dict['pose_body'].astype(np.float32))
            smpl_data['pose_hand'].append(model_dict['pose_hand'].astype(np.float32))
            smpl_data['trans'].append(model_dict['trans'].astype(np.float32))
        # betas转为tensor


        return smpl_data

    def __len__(self):
        return len(self.data)
    
    # 修改数据读取逻辑（关键优化点）：
    def getitem(self, idx, data_dict=None):

        start = time.time()
        device = self.device

        if data_dict is None:
            data_dict = self.data[idx]
        cam_name = data_dict['cam_name']
        frame_idx = data_dict['frame_idx']
        model_file = data_dict['model_file']
        # 将CPU密集型操作提前并合并：
        model_dict = np.load(model_file, mmap_mode='r')  # 使用内存映射减少IO时间
        
        # 将以下操作批量转换为Tensor后一次性传输到GPU：
        smpl_params = {
            'trans': model_dict['trans'],
            'root_orient': model_dict['root_orient'],
            'pose_body': model_dict['pose_body'],
            'pose_hand': model_dict['pose_hand'],
            'betas': model_dict['betas']
        }
        # 修改数据传输逻辑：
        smpl_tensors = {
            k: torch.from_numpy(v).pin_memory().to(device, non_blocking=True)
            for k, v in smpl_params.items()
        }
    
        # 合并矩阵运算（替代多次单独操作）
        pose = torch.cat([smpl_tensors['root_orient'], 
                        smpl_tensors['pose_body'], 
                        smpl_tensors['pose_hand']], dim=-1).reshape(-1, 3)
        
        trans = smpl_tensors['trans']

        # === 读取相机参数（保持在 CPU）
        K = np.array(self.cameras[cam_name]['K'], dtype=np.float32).copy()
        R = np.array(self.cameras[cam_name]['R'], np.float32)
        T = np.array(self.cameras[cam_name]['T'], np.float32)
        T = T[:, 0]
        
        # === 焦距计算（仍然在 CPU）
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
        # === bone_transform 校准并添加 trans
        bone_transforms = bone_transforms @ torch.linalg.inv(bone_transforms_02v)

        bone_transforms[:, :3, 3] += trans

        # === pose_rot 组装
        pose_mat = pose_mat_full[1:]  # 23 x 3 x 3
        pose_rot = torch.cat([torch.eye(3, device=device).unsqueeze(0), pose_mat], dim=0).reshape(-1, 9).unsqueeze(0)
        end = time.time()
        print("getitem time:", end - start)
        # === Camera 封装（你原来的类）
        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image=None, mask=None,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=device,
            rots=pose_rot,
            Jtrs=Jtr_norm.unsqueeze(0),
            bone_transforms=bone_transforms,
            smplth=self.smplth,
            height=self.h, weight=self.w,
            Height=self.H, Weight=self.W,
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    def readPointCloud(self,):
        if self.cfg.get('random_init',False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 63_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                face_weights = np.mean(self.skinning_weights['neutral'][faces], axis=1)
                n_points_hand = 7_000
                n_points_body = 53_000

                # 计算手部面权重
                hand_joint_indices = [
                    20, 21, 37, 38, 39, 25, 26, 27, 34, 35, 36, 31, 32, 33, 28, 29, 30, 
                    52, 53, 54, 40, 41, 42, 49, 50, 51, 46, 47, 48, 43, 44, 45
                ]
                hand_weights = face_weights[:, hand_joint_indices].sum(axis=1)
                threshold = 0.05
                hand_faces = np.where(hand_weights > threshold)[0]
                body_faces = np.where(hand_weights <= threshold)[0]

                verts0 = verts[faces[:, 0], :]
                verts1 = verts[faces[:, 1], :]
                verts2 = verts[faces[:, 2], :]

                # # 计算面面积和面权重
                # face_areas = 0.5 * np.linalg.norm(np.cross(verts1 - verts0, verts2 - verts0), axis=1)
                # facetoweights = face_areas / face_areas.sum()

                # body_face_weights = facetoweights[body_faces]

                # 分别为手部和身体区域采样点
                hand_points = mesh.sample(
                    n_points_hand,
                    face_weight=np.isin(np.arange(len(faces)), hand_faces).astype(float)
                )   
                body_points = mesh.sample(
                    n_points_body,
                )

                # 合并手部和身体区域的采样点
                xyz = np.vstack((hand_points, body_points))
                # 设置点的颜色：手部为红色，非手部为白色
                rgb = np.ones_like(xyz) * 255
                rgb[:n_points_hand] = [255, 0, 0]  # 手部为红色
                rgb[n_points_hand:] = [255, 255, 255]  # 非手部为白色

                storePly(ply_path, xyz, rgb)
                pcd = fetchPly(ply_path)

        return pcd
