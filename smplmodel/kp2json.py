import numpy as np
import json
import pickle
from torch import Tensor

with open('/home/lyh/pythonproject/EasyMocap-master-new/IMG_3233.pkl', 'rb') as f:
    data = pickle.load(f)

for idx in range(0, 737):
    print(f"Processing index: {idx}")

    if len(data) > idx and len(data[idx]) > 0:
        try:
            Rh = np.array(Tensor.cpu(data[idx][0]['rotvec'])[0]).reshape([1, 3]).astype(np.float32)
            print(f"Rh at index {idx}: {Rh}")

            Th = np.array(Tensor.cpu(data[idx][0]['transl_pelvis']))

            shapes = np.array([[-0.7631, -0.1239,  0.0475, -0.2054, -0.1537, -0.1305, -0.0073,  0.0860,
        -0.1498,  0.1450]])
            expression = np.array(Tensor.cpu(data[idx][0]['expression'])).reshape([1,10]).astype(np.float32)
            pose_body = np.array(Tensor.cpu(data[idx][0]['rotvec']))[1:22].reshape([1, 63]).astype(np.float32)
            pose_jaw = np.array(Tensor.cpu(data[idx][0]['rotvec']))[52:].reshape([1, 3]).astype(np.float32)
            pose_hands = np.array(Tensor.cpu(data[idx][0]['rotvec']))[22:52].reshape([1, 90]).astype(np.float32)
            zeros = np.zeros(3).reshape([1, -1])
            poses = np.concatenate([zeros, pose_body,pose_jaw,zeros,zeros,pose_hands], axis=1).astype(np.float32)

            json_data = [{
                'Th': Th.tolist(),
                'Rh': Rh.tolist(),
                'poses': poses.tolist(),
                'shapes': shapes.tolist(),
                'expression': expression.tolist()

            }]

            json_file = f'/home/lyh/pythonproject/EasyMocap-master-new/lyh/smplx/{idx:06d}.json'
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=4)

        except IndexError as e:
            print(f"IndexError at idx {idx}: {e}")
        except KeyError as e:
            print(f"KeyError at idx {idx}: {e}")
    else:
        print(f"Data not available for idx {idx}")