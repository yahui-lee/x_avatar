import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d.ops as ops
import trimesh
import igl

from utils.general_utils import build_rotation, rotation_matrix_to_quaternion
from models.network_utils import get_skinning_mlp

class RigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera):
        raise NotImplementedError

    def regularization(self):
        return NotImplementedError

class Identity(RigidDeform):
    """ Identity mapping for single frame reconstruction """
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera):
        return gaussians

    def regularization(self):
        return {}

class SMPLNN(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def forward(self, gaussians, iteration, camera):
        bone_transforms = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        pts_W = self.query_weights(xyz)
        T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self):
        return {}

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return F.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_point, n_dim = x.shape

    prob_all = torch.ones(n_point, 55, device=x.device)
    # softmax_x = F.softmax(x, dim=-1)
    sigmoid_x = sigmoid(x).float()

    prob_all[:, [1, 2, 3]] = sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = 1 - sigmoid_x[:, [0]]
    # print('soft',sigmoid_x[:, [0]].shape,softmax(x[:, [1, 2, 3]]).shape,prob_all[:, [1, 2, 3]].shape)
    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid_x[:, [4, 5, 6]])
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid_x[:, [7, 8, 9]])
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid_x[:, [10, 11]])
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [55]] * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [55]])

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid_x[:, [15]])
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid_x[:, [16, 17]])
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid_x[:, [18, 19]])
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid_x[:, [20, 21]])
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

    # if smpl_type == 'smpl':
    #     prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid_x[:, [22, 23]])
    #     prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid_x[:, [22, 23]])

    prob_all[:, [22, 23,24]] = prob_all[:, [15]] * sigmoid_x[:, [56]] *softmax(x[:, [22,23,24]])
    prob_all[:, [15]] = prob_all[:, [15]] * (1 - sigmoid_x[:, [56]])   

    prob_all[:,[25,28,31,34,37]] = prob_all[:,[20]]* sigmoid_x[:, [57]]*softmax(x[:, [25,28,31,34,37]])
    prob_all[:,[20]] = prob_all[:, [20]] * (1 - sigmoid_x[:, [57]])  

    prob_all[:,[40,43,46,49,52]] = prob_all[:,[21]]* sigmoid_x[:, [58]]*softmax(x[:, [40,43,46,49,52]])
    prob_all[:,[21]] = prob_all[:, [21]] * (1 - sigmoid_x[:, [58]])  

    prob_all[:,[26,29,32,35,38]] = prob_all[:,[25,28,31,34,37]]* (sigmoid_x[:, [26,29,32,35,38]])
    prob_all[:,[25,28,31,34,37]] = prob_all[:, [25,28,31,34,37]] * (1 - sigmoid_x[:, [26,29,32,35,38]])

    prob_all[:,[27,30,33,36,39]] = prob_all[:,[26,29,32,35,38]]* (sigmoid_x[:,[27,30,33,36,39]])
    prob_all[:,[26,29,32,35,38]] = prob_all[:, [26,29,32,35,38]] * (1 - sigmoid_x[:,[27,30,33,36,39]])

    prob_all[:,[41,44,47,50,53]] = prob_all[:,[40,43,46,49,52]]* (sigmoid_x[:, [41,44,47,50,53]])
    prob_all[:,[40,43,46,49,52]] = prob_all[:, [40,43,46,49,52]] * (1 - sigmoid_x[:, [41,44,47,50,53]])

    prob_all[:,[42,45,48,51,54]] = prob_all[:,[41,44,47,50,53]]* (sigmoid_x[:, [42,45,48,51,54]])
    prob_all[:,[41,44,47,50,53]] = prob_all[:, [41,44,47,50,53]] * (1 - sigmoid_x[:, [42,45,48,51,54]])


    return prob_all

class SkinningField(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_type = cfg.smpl_type
        self.smpl_verts = metadata["smpl_verts"]
        self.skinning_weights = metadata["skinning_weights"]
        self.aabb = metadata["aabb"]
        self.cano_mesh = metadata["cano_mesh"]
        if self.smpl_type == 'smpl':
            self.faces = np.load('body_models/misc/faces.npz')['faces']
        elif self.smpl_type == 'smplx':
            self.faces = np.load('body_models/miscx/faces.npz')['faces']
        else:
            raise ValueError
        

        # 蒸馏 体素网格
        self.distill = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()
        # if self.smpl_type == 'smplx':
        #     self.d_out = 56
        self.lbs_network = get_skinning_mlp(3, cfg.d_out, cfg.skinning_network)


    def precompute(self, recompute_skinning=True):
        if recompute_skinning or not hasattr(self, "lbs_voxel_final"):
            d, h, w = self.resolution

            lbs_voxel_final = self.lbs_network(self.grid[0]).float()
            lbs_voxel_final = self.cfg.soft_blend * lbs_voxel_final

            lbs_voxel_final = self.softmax(lbs_voxel_final)

            self.lbs_voxel_final = lbs_voxel_final.permute(1, 0).reshape(1, self.d_out-1, d, h, w)

    def get_forward_transform(self, xyz, tfs):
        if self.distill:
            self.precompute(recompute_skinning=self.training)
            fwd_grid = torch.einsum("bcdhw,bcxy->bxydhw", self.lbs_voxel_final, tfs[None])
            fwd_grid = fwd_grid.reshape(1, -1, *self.resolution)
            T_fwd = F.grid_sample(fwd_grid, xyz.reshape(1, 1, 1, -1, 3), padding_mode='border')
            T_fwd = T_fwd.reshape(4, 4, -1).permute(2, 0, 1)
        else:
            pts_W = self.lbs_network(xyz)
            pts_W = self.softmax(pts_W)
            T_fwd = torch.matmul(pts_W, tfs.view(-1, 16)).view(-1, 4, 4).float()
        return T_fwd

    def sample_skinning_loss(self):
        face_weights = np.mean(self.skinning_weights[self.cano_mesh.faces], axis=1)

        hand_joint_indices = [
        37, 38, 39, 25, 26, 27, 34, 35, 36, 31, 32, 33, 28, 29, 30, 
        52, 53, 54, 40, 41, 42, 49, 50, 51, 46, 47, 48, 43, 44, 45 
        ]

        
        hand_weights = face_weights[:, hand_joint_indices].sum(axis=1)
        
        threshold = 0.05
        hand_faces = np.where(hand_weights > threshold)[0]
        body_faces = np.where(hand_weights <= threshold)[0]

        verts0 = self.smpl_verts[self.faces[:, 0], :]
        verts1 = self.smpl_verts[self.faces[:, 1], :]
        verts2 = self.smpl_verts[self.faces[:, 2], :]

        
        face_areas = 0.5 * np.linalg.norm(np.cross(verts1 - verts0, verts2 - verts0), axis=1)
        face_weights = face_areas / face_areas.sum()

        # body_face_weights = face_weights[body_faces]

        hand_points, hand_face_idx = self.cano_mesh.sample(
            self.cfg.hand_reg_pts,
            face_weight=np.isin(np.arange(len(self.faces)), hand_faces).astype(float),
            return_index=True,
        )   
        body_points, body_face_idx = self.cano_mesh.sample(
            self.cfg.n_reg_pts,
            return_index=True,
        )
        all_points = []
        all_weights = []

        # breakpoint()
        for point_skinning, face_idx in [(hand_points, hand_face_idx), (body_points, body_face_idx)]:
            point_skinning = point_skinning.view(np.ndarray).astype(np.float32)
            bary_coords = igl.barycentric_coordinates_tri(
                point_skinning.astype(np.float32),
                self.smpl_verts[self.faces[face_idx, 0], :].astype(np.float32),
                self.smpl_verts[self.faces[face_idx, 1], :].astype(np.float32),
                self.smpl_verts[self.faces[face_idx, 2], :].astype(np.float32),
            )
            vert_ids = self.faces[face_idx, ...]
            pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)

            point_skinning = torch.from_numpy(point_skinning).cuda()
            pts_W = torch.from_numpy(pts_W).cuda()

            all_points.append(point_skinning)
            all_weights.append(pts_W)
        
        all_points = torch.cat(all_points, dim=0)
        all_weights = torch.cat(all_weights, dim=0)

        return all_points, all_weights
    
    def softmax(self, logit):
        if logit.shape[-1] == 59:
            w = hierarchical_softmax(logit)
        elif logit.shape[-1] == 55:
            w = F.softmax(logit, dim=-1)
        else:
            raise ValueError
        return w

    def get_skinning_loss(self):
        pts_skinning, sampled_weights = self.sample_skinning_loss()
        pts_skinning = self.aabb.normalize(pts_skinning, sym=True)

        if self.distill:
            pred_weights = F.grid_sample(self.lbs_voxel_final, pts_skinning.reshape(1, 1, 1, -1, 3), padding_mode='border')
            pred_weights = pred_weights.reshape(55, -1).permute(1, 0)
        else:
            pred_weights = self.lbs_network(pts_skinning)
            pred_weights = self.softmax(pred_weights)
        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        np.set_printoptions(threshold=np.inf)
        # print('采样权重：',sampled_weights[0].cpu().detach().numpy())
        # print('预测权重：',pred_weights[0].cpu().detach().numpy())
        # print('权重损失：',skinning_loss)
        # breakpoint()

        return skinning_loss


    def forward(self, gaussians, iteration, camera):
        tfs = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        T_fwd = self.get_forward_transform(xyz_norm, tfs)
        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        #deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self):
        loss_skinning = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning
        }

def get_rigid_deform(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "smpl_nn": SMPLNN,
        "skinning_field": SkinningField,
    }
    return model_dict[name](cfg, metadata)


# def compute_joint_positions_relative_to_root(rel_transforms, root_position):
#     joint_positions = []

#     for i in range(rel_transforms.shape[0]):
#         joint_position = rel_transforms[i] @ root_position
#         joint_positions.append(joint_position[:3].cpu().numpy())
        
#     return joint_positions

