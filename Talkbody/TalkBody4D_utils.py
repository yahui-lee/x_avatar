import numpy as np
import torch
import json, os, argparse
from tqdm import tqdm
from smplx.body_models import create
import cv2

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    axis = axis_angle / (angles + 1e-6)  # Avoid division by zero
    x, y, z = torch.unbind(axis, dim=-1)

    sin_theta = torch.sin(angles)
    cos_theta = torch.cos(angles)
    one_minus_cos_theta = 1 - cos_theta

    o = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([o, -z, y], dim=-1),
            torch.stack([z, o, -x], dim=-1),
            torch.stack([-y, x, o], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    R = (
        eye
        + sin_theta.unsqueeze(-1) * K
        + one_minus_cos_theta.unsqueeze(-1) * torch.matmul(K, K)
    )

    return R

def read_json(fpath):
    with open(fpath,'r') as f:
        obj = json.load(f)
    return obj

def read_cameras(calib_path):
    
    calib = read_json(calib_path)

    cameras = {}
    for cam_ID in calib['cameras']:
        image_size = calib['cameras'][cam_ID]['image_size']
        K = np.array(calib['cameras'][cam_ID]['K']).astype(np.float32)
        dist = np.array(calib['cameras'][cam_ID]['dist']).astype(np.float32).reshape(-1,)
        cameras[cam_ID] = {
            'K': K,
            'dist': dist,
            'width': image_size[0],
            'height': image_size[1],
        }

    for cam_ID_ in calib['camera_poses']:
        if cam_ID_.find('to') != -1:
            cam_ID = cam_ID_[:cam_ID_.find('to')-1]
        else:
            cam_ID = cam_ID_
        R = np.array(calib['camera_poses'][cam_ID_]['R']).astype(np.float32)
        T = np.array(calib['camera_poses'][cam_ID_]['T']).astype(np.float32)
        cameras[cam_ID].update({
            'R': R,
            'T': T.reshape(3,),
        })

    return cameras


def main(args):

    body_model = create(model_path=args.model_path, model_type=args.model_type, gender=args.gender, num_betas=args.num_betas, num_expression_coeffs=args.num_expression_coeffs, use_pca=args.use_pca, num_pca_comps=args.num_pca_comps, flat_hand_mean=args.flat_hand_mean, ext=args.ext)

    cameras = read_cameras(args.calib_file)

    faces = body_model.faces.astype(np.int32)

    cam_IDs = list(cameras.keys())

    frame_IDs = os.listdir(args.params_folder)
    frame_IDs = [frame_ID[:-5] for frame_ID in frame_IDs if frame_ID.endswith('.json')]
    frame_IDs.sort()

    frame_IDs = frame_IDs[::40]

    os.makedirs(args.output_folder, exist_ok=True)

    for frame_ID in tqdm(frame_IDs, total=len(frame_IDs)):

        params_path = os.path.join(args.params_folder, '{}.json'.format(frame_ID))
        params = read_json(params_path)

        with torch.no_grad():
            body_params = {}
            for param_name in params:
                body_params[param_name] = torch.tensor(params[param_name]).float()
                if param_name == 'betas':
                    body_params[param_name] = body_params[param_name][:, :args.num_betas]
                if param_name == 'expression':
                    body_params[param_name] = body_params[param_name][:, :args.num_expression_coeffs]
            smpl_out = body_model(**body_params)
            verts = smpl_out['vertices']
            Rh = body_params['Rh'] # (1, 3)
            Th = body_params['Th'] # (1, 3)
            Rh = axis_angle_to_matrix(Rh)
            verts = verts @ Rh.transpose(1, 2) + Th.unsqueeze(1)
            verts = verts.squeeze(0).numpy()

        if args.save_mesh:
            mesh_file = open(os.path.join(args.output_folder, '{}.obj'.format(frame_ID)), 'w')
            for v in verts:
                mesh_file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
            for f in faces:
                f_plus = f + 1
                mesh_file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
            mesh_file.close()

        for cam_ID in cam_IDs[0:1]:
            img_fpath = os.path.join(args.images_folder, cam_ID, '{}.jpg'.format(frame_ID))
            img = cv2.imread(img_fpath, cv2.IMREAD_UNCHANGED)

            camera = cameras[cam_ID]

            K = camera['K'].reshape(3, 3)
            R = camera['R'].reshape(3, 3)
            T = camera['T'].reshape(3, 1)
            img_w, img_h = camera['width'], camera['height']

            dist = camera['dist'].reshape(-1, )

            img = cv2.undistort(img, K, dist)

            smplx_verts_cam = np.matmul(R, verts.T) + T

            smplx_verts_proj = np.matmul(K, smplx_verts_cam)

            smplx_verts_proj /= smplx_verts_proj[2, :]
            smplx_verts_proj = smplx_verts_proj[:2, :].T

            smplx_verts_proj = np.round(smplx_verts_proj).astype(np.int32)
            smplx_verts_proj[:, 0] = np.clip(smplx_verts_proj[:, 0], 0, img.shape[1] - 1)
            smplx_verts_proj[:, 1] = np.clip(smplx_verts_proj[:, 1], 0, img.shape[0] - 1)

            for v in smplx_verts_proj:
                img[v[1], v[0], :] = np.array([255, 255, 255], dtype = np.uint8)

            cv2.imwrite(os.path.join(args.output_folder, '{}.jpg'.format(frame_ID)), img)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, required=True, help='path of input images')
    parser.add_argument('--params_folder', type=str, required=True,
                        help='the model params folder')
    parser.add_argument('--calib_file', type=str, required=True, help='path of input calibration')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='the output folder')
    parser.add_argument('--model_path', type=str, default='assets/smplx_models',
                        help='model path of smplx')
    parser.add_argument('--model_type', type=str, default='smplx',
                        help='smplx models type')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='gender')
    parser.add_argument('--num_betas', type=int, default=100)
    parser.add_argument('--num_expression_coeffs', type=int, default=50)
    parser.add_argument('--use_pca', action='store_true')
    parser.add_argument('--num_pca_comps', type=int, default=6)
    parser.add_argument('--flat_hand_mean', action='store_true', default=True)
    parser.add_argument('--ext', type=str, default='npz')
    parser.add_argument('--save_mesh', action='store_true', default=False,
                        help='if save mesh obj file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    main(args)