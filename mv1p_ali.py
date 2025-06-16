from smplmodel import load_model
import json
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import matplotlib.pyplot as plt
# 请将/smplx/SMPLX_{性别}.pkl放在此目录下(代码会读取路径：'path/smplx/SMPLX_{}.pkl')，文件可从smplx官网下载
model_path = 'path'  # smplx
model_gender = 'MALE'
#可以自行添加维度，SMPLX最高支持300维shape，100维expression
body_model = load_model(gender=model_gender, model_path=model_path, num_shape_coeffs = 100, num_expression_coeffs = 50)

def get_minimal_shape(t_params):
    body_params = dict()
    for key, val in t_params[0].items():
        if key == 'shapes':
            val = torch.Tensor(val)
            body_params[key] = val.view(1, -1)
        else:
            val = torch.Tensor(val)
            body_params[key] = torch.zeros([1, len(val.reshape(-1))])
    minimal_shape, _ = body_model(return_verts=True, return_tensor=True, return_smpl_joints=False, **body_params)
    #verts,_ = body_model(return_verts=False, return_tensor=True, return_smpl_joints=True, **body_params)
    # LYH3
    #diff = verts.squeeze()[0]-(verts.squeeze()[13]+verts.squeeze()[14])/2
    # LYH4
    #diff = verts.squeeze()[0] - verts.squeeze()[12]  + np.array([0.01, 0.04, 1.1]).reshape([1,3]).astype(np.float32)
    # minimal_shape[0] = minimal_shape[0] - (verts.squeeze()[14]+verts.squeeze()[13])/2
    # print(minimal_shape[0])
    # v1 = verts[0].numpy()
    # j1 = minimal_shape[0].numpy()

    # xv = [k[0] for k in v1]
    # yv = [k[1] for k in v1]
    # zv = [k[2] for k in v1]
    # xj = [k[0] for k in j1]
    # yj = [k[1] for k in j1]
    # zj = [k[2] for k in j1]
    # fig1 = plt.figure(dpi=200)
    # ax = fig1.add_subplot(111, projection='3d')
    # plt.title('point cloud')
    # ax.scatter(xv, yv, zv,c='b',marker='.',s=20, linewidth=0, alpha=1,cmap='spectral')
    # ax.scatter(xj, yj, zj, c='r', marker='*', s=20, linewidth=0, alpha=1, cmap='spectral')
    # ax.set_xlabel('x label')
    # ax.set_ylabel('y label')
    # ax.set_zlabel('z label')
    # plt.show()
    return minimal_shape

def get_kpts_3dvert(t_params):
    body_params = dict()
    for key, val in t_params[0].items():
        if key == 'id': continue
        val = torch.Tensor(val)
        body_params[key] = val.view(1, -1)
    kpts_3dvert, bone_transforms = body_model(return_verts=False, return_tensor=True, return_smpl_joints=True, **body_params)
    return kpts_3dvert, bone_transforms

if __name__ == '__main__':
    for i in range(1280):
        smplx_path = f'/home/lyh/pythonproject/EasyMocap-master-new/UseSmplDemo/smplx/{i:06d}.json'
        with open(smplx_path, 'r') as f:
            t_params = json.load(f)
        minimal_shape = get_minimal_shape(t_params)
        # faces = np.load('/home/lyh/pythonproject/xavatar/body_models/miscx/faces.npz')['faces']
        # vertices = np.array(minimal_shape).squeeze().reshape((10475,3)).astype(np.float32)
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # out_filename = os.path.join('.', '{:06d}.ply'.format(i))
        # mesh.export(out_filename)
        minimal_shape = np.array(minimal_shape).squeeze().reshape((10475,3)).astype(np.float32)
        
        # print(bone_transforms)
        betas = np.array(t_params[0]['shapes'], dtype=np.float32)
        #trans = np.array(t_params[0]['Th']).reshape([1,3]).astype(np.float32) - np.array(diff).reshape([1,3]).astype(np.float32)
        trans = np.array(t_params[0]['Th']).reshape([1,3]).astype(np.float32)
        t_params[0]['Th'] = trans.tolist()
        root_orient = np.array(t_params[0]['Rh']).reshape([1,3]).astype(np.float32)
        expression = np.array(t_params[0]['expression']).reshape([1,-1]).astype(np.float32)

        kpts_3dvert, bone_transforms = get_kpts_3dvert(t_params)
        kpts_3dvert = np.array(kpts_3dvert).squeeze().reshape((55,3)).astype(np.float32)
        bone_transforms = np.array(bone_transforms).squeeze().reshape((55,4,4)).astype(np.float32)

        poses = np.array(t_params[0]['poses'], dtype=np.float32)
        pose_body = poses[:, 3:66]
        pose_hand = poses[:, 66:]
        print(trans[0], i)
        out_filename = os.path.join('/home/lyh/pythonproject/EasyMocap-master-new/UseSmplDemo/test', '{:06d}.npz'.format(i))
        np.savez(out_filename,
                  minimal_shape=minimal_shape,
                  betas = betas,
                  Jtr_posed = kpts_3dvert,
                  bone_transforms=bone_transforms,
                  trans = trans[0],
                  root_orient = root_orient[0],
                  pose_body = pose_body[0],
                  pose_hand = pose_hand[0],
                  expression = expression[0]
                  )



