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

def main(args):

    body_model = create(model_path=args.model_path, model_type=args.model_type, gender=args.gender, num_betas=args.num_betas, num_expression_coeffs=args.num_expression_coeffs, use_pca=args.use_pca, num_pca_comps=args.num_pca_comps, flat_hand_mean=args.flat_hand_mean, ext=args.ext)



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
                if param_name == 'v_shape':
                    body_params[param_name] = None
                    continue
                print(param_name,body_params[param_name].shape)
            # 方式 1：手动 Rh + Th（作为对照）
            smpl_out1 = body_model(**body_params)
            verts1 = smpl_out1['vertices']
            Rh_mat = axis_angle_to_matrix(body_params['Rh'])  # (1, 3, 3)
            Th = body_params['Th']  # (1, 3)
            verts1 = verts1 @ Rh_mat.transpose(1, 2) + Th.unsqueeze(1)
            verts1 = verts1.squeeze(0).cpu().numpy()

            # 方式 2：直接设置 global_orient 和 transl，移除 Rh / Th
            body_params2 = body_params.copy()
            body_params2['global_orient'] = body_params2.pop('Rh')
            body_params2['transl'] = body_params2.pop('Th')
            smpl_out2 = body_model(**body_params2)
            verts2 = smpl_out2['vertices'].squeeze(0).cpu().numpy()

            # 差值
            diff = verts1 - verts2  # shape: (N, 3)

            # 所有点的差是否都一致？
            is_constant_shift = np.allclose(diff, diff[0], atol=1e-3)
            # 假设使用 global_orient + transl（方式2）是你的最终目标

            if is_constant_shift:
                t_fix = diff[0]  # 就是 verts1 - verts2 的平移向量
                print("✅ 应用平移修正向量:", t_fix)

                # 修正 transl：
                body_params2['transl'] += torch.tensor(t_fix).unsqueeze(0)
                smpl_out_fixed = body_model(**body_params2)
                verts_fixed = smpl_out_fixed['vertices'].squeeze(0).cpu().numpy()

                # 再次验证误差
                error_fix = np.linalg.norm(verts1 - verts_fixed, axis=1)
                print(f"[{frame_ID}] 修正后 max error: {error_fix.max():.6f}, mean error: {error_fix.mean():.6f}")

                # 这个 verts_fixed 就是你想要的了
                verts = verts_fixed



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, default='/home/lyh/pythonproject/multi-hmr/Talkbody/data/images', help='path of input images')
    parser.add_argument('--params_folder', type=str, default='/home/lyh/pythonproject/multi-hmr/Talkbody/data/smplx_fitting',
                        help='the model params folder')
    parser.add_argument('--calib_file', type=str, default='/home/lyh/pythonproject/multi-hmr/Talkbody/data/calibration.json', help='path of input calibration')
    parser.add_argument('--output_folder', type=str, default='/home/lyh/pythonproject/multi-hmr/Talkbody/out',
                        help='the output folder')
    parser.add_argument('--model_path', type=str, default='/home/lyh/pythonproject/multi-hmr/models',
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

