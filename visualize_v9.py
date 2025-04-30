from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import sys

import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from scene import Scene,GaussianModel
import socket
import cv2
from tqdm import trange
from gaussian_rendererforview import render
from utils.general_utils import fix_random
import hydra
from omegaconf import OmegaConf
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal ,getProjectionMatrix
from datetime import datetime
import time
import threading
import ast
import re
from queue import Queue
from collections import deque
def parse_tensor_string(s):
    # 1. 清理 device 和 dtype
    s = re.sub(r",?\s*device='[^']*'", '', s)
    s = re.sub(r",?\s*dtype=torch\.\w+", '', s)

    # 2. 提取 tensor(...) 内容
    tensor_strs = re.findall(r'tensor\((\[.*?\]|\(.*?\))\)', s, re.DOTALL)
    tensor_list = []

    for ts in tensor_strs:
        try:
            data = ast.literal_eval(ts)
            t = torch.tensor(data)
            tensor_list.append(t)
        except Exception as e:
            print(f"[Parse error] Failed to parse tensor from:\n{ts}\nError: {e}")
    
    return tensor_list

def move_camera( R, T, trans, horizontal_angle, vertical_angle, rotate_axis='y', inv_angle=False):
    Ri = np.array(R, np.float32)
    Ti = np.array(T, np.float32)
    Ei = np.eye(4)
    Ei[:3,:3] = Ri
    Ei[:3,3:] = Ti.reshape((3, 1))
    angle_x = np.deg2rad(vertical_angle)
    angle_y = np.deg2rad(horizontal_angle)

    rot_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]], dtype=np.float32)

    rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=np.float32)

    rotation_matrix = rot_y @ rot_x
    if inv_angle:
        angle = -angle
        
    inv_E = np.linalg.inv(Ei)
    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle

    # rotate_coord = {'x':0, 'y':1, 'z':2}
    # grot_vec = np.array([0., 0., 0.])
    # grot_vec[rotate_coord[rotate_axis]] = angle
    # grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = rotation_matrix.dot(campos)
    rot_camrot = rotation_matrix.dot(camrot)
    if trans is not None:
        rot_campos += trans

    return(rot_camrot.T, -rot_camrot.T.dot(rot_campos))


class MainWindow(QMainWindow):
    def __init__(self, hw):
        super().__init__()
        self.initUI()
        self.times = 0
        self.render_timer = QTimer(self)
        self.last_x, self.last_y = None, None
        self.horizontal_angle = 0
        self.vertical_angle = 0
        self.imgQueue = Queue(maxsize=300)
        self.renderQueue = Queue(maxsize=300)
        self.smplQueue = Queue(maxsize=5000)
        self.frame_times = deque(maxlen=60)
        self.h, self.w = hw[0], hw[1]

    def initUI(self):
        self.statusLabel = QLabel('times: 0', self)
        self.statusLabel.setGeometry(10, 10, 200, 30)
        self.imageLabel = QLabel(self)
        self.imageLabel.setGeometry(10, 50, 1280, 1280)
        self.setFixedSize(1120, 1320)
        self.setWindowTitle('3DGS Rendering with PyQt5')

        # left button
        self.leftButton = QPushButton('Left', self)
        self.leftButton.setGeometry(170, 10, 200, 30)
        self.leftButton.clicked.connect(self.move_camera_left)
        
        # right button
        self.rightButton = QPushButton('Right', self)
        self.rightButton.setGeometry(380, 10, 200, 30)
        self.rightButton.clicked.connect(self.move_camera_right)

        self.upButton = QPushButton('Up', self)
        self.upButton.setGeometry(590, 10, 200, 30)
        self.upButton.clicked.connect(self.move_camera_up)

        self.downButton = QPushButton('Down', self)
        self.downButton.setGeometry(800, 10, 200, 30)
        self.downButton.clicked.connect(self.move_camera_down)

    def update_image(self): 
        self.render_timer.start(20)
        self.render_timer.timeout.connect(self.update_frame)
        
        # if t1.is_alive():
        #     time.sleep(0.0001)


    def update_frame(self, ):
        if not self.imgQueue.empty():
            self.image = self.imgQueue.get()

            # 记录刷新时间点
            now = time.time()
            self.frame_times.append(now)

            # 计算平均 FPS
            fps = 0
            if len(self.frame_times) >= 2:
                fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

            self.times += 1
            print(f"Updating frame:{self.times}  FPS: {fps:.2f}")
            

            q_image = QImage(self.image.astype(np.uint8).data.tobytes(), self.w, self.h, 3 * self.w, QImage.Format.Format_RGB888).copy()
            q_imagescaled = q_image.scaled(1080, 1280, Qt.AspectRatioMode.KeepAspectRatio)
            pixmap = QPixmap.fromImage(q_imagescaled)
            
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.repaint() 
            
            self.statusLabel.setText(f'times: {self.times}  FPS: {fps:.2f}')

        

    def move_camera_left(self):
        print("Moving camera left")
        self.horizontal_angle -= 1

    def move_camera_right(self):
        print("Moving camera right")
        self.horizontal_angle += 1

    def move_camera_up(self):
        print("Moving camera up")
        self.vertical_angle += 1

    def move_camera_down(self):
        print("Moving camera down")
        self.vertical_angle -= 1

    def mousePressEvent(self, event):
        # 记录鼠标按下的位置
        if event.button() == Qt.LeftButton:
            self.last_x, self.last_y = event.x(), event.y()

    def mouseMoveEvent(self, event):
        # 根据鼠标移动来旋转相机
        if self.last_x is not None and self.last_y is not None:
            dx = event.x() - self.last_x
            dy = event.y() - self.last_y
            self.last_x, self.last_y = event.x(), event.y()
            self.rotate_camera(dx, dy)

    def rotate_camera(self, dx, dy):
        self.horizontal_angle += dx / 5.0  # 根据需要调整灵敏度
        self.vertical_angle += dy / 5.0  # 根据需要调整灵敏度
        self.horizontal_angle = self.horizontal_angle % 360  # 保持角度在360度内
        self.vertical_angle = max(min(self.vertical_angle, 90), -90)  # 限制垂直角度在-90到90度之间

    def get_angles(self):
        return self.horizontal_angle, self.vertical_angle


def renderimg(config, scene, get_angles, smplQueue, renderQueue):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        while True:
            gettime = time.time()
            data_dict = smplQueue.get()

            print("接受一帧：", smplQueue.qsize(),time.time() - gettime)
            view = scene.test_dataset.getitem(0, data_dict)
        
            horizontal_angle, vertical_angle = get_angles()
            if horizontal_angle != 0 or vertical_angle != 0:
                # update camera parameters
                view.R, view.T = move_camera(view.R, view.T, view.smplth, horizontal_angle, -vertical_angle)
            view.T = view.T.reshape((3, 1))
            M = np.eye(3)
            M[0, 2] = (view.K[0, 2] - view.original_width / 2) / view.K[0, 0]
            M[1, 2] = (view.K[1, 2] - view.original_height / 2) / view.K[1, 1]
            view.K[0, 2] = view.original_width / 2
            view.K[1, 2] = view.original_height / 2
            view.R = M @ view.R
            view.T = M @ view.T
            view.R = np.transpose(view.R)
            view.T = view.T[:, 0]
            
            view.K[0, :] *= view.image_width / view.original_width
            view.K[1, :] *= view.image_height / view.original_height
            view.FoVy = focal2fov(view.K[1, 1], view.image_height) * 1.1
            view.FoVx = focal2fov(view.K[0, 0], view.image_width) * 1.1
            view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, 1.0)).transpose(0, 1).cuda()
            view.projection_matrix  = getProjectionMatrix(znear=0.01, zfar=100, fovX=view.FoVx,
                                                          fovY=view.FoVy).transpose(0, 1).cuda()
            view.full_proj_transform = (
                view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.data['camera_center'] = view.world_view_transform.inverse()[3, :3]
            
            pc, loss_reg, colors_precomp = scene.convert_gaussians(view, 30000, False)
            renderQueue.put((view, config.pipeline, background, pc, colors_precomp, False, False))
            
            time.sleep(0.0001)

def gsrender(renderQueue, imgQueue):
    with torch.set_grad_enabled(False):
        while True:
            try:
                start = time.time()
                view, pipeline, background, pc, colors_precomp, compute_loss, return_opacity = renderQueue.get()

                print("渲染耗时：", time.time() - start)
                rendering = render(view, pipeline, background, pc, colors_precomp, compute_loss = compute_loss,  return_opacity = return_opacity)
                
                imgQueue.put(np.transpose(rendering.cpu().numpy(),(1,2,0))*255)
                
                if imgQueue.qsize() > 3000:
                    time.sleep(imgQueue.qsize()/30)

                print(imgQueue.qsize())
                time.sleep(0.0001)
            except Exception as e:
                print("gsrender出错:", e)


def getsmpl_net(host, port, queue):
    # 创建UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"[SMPL] Listening on {host}:{port}...")
    idx = -1
    while True:
        get = time.time()
        data, addr = sock.recvfrom(65535)

        tensors = parse_tensor_string(data.decode('utf-8'))
        if len(tensors) != 3:
            print("[Error] SMPL tensor数目错误")
            continue

        transl_pelvis, rotvec, shape = tensors
        trans = transl_pelvis.view(3)
        root_orient = rotvec[0, :3].view(3)
        pose_body = rotvec[1:22, :3].contiguous().view(-1)
        jaw = rotvec[-1, :3]
        pose_body = torch.cat([pose_body, jaw, torch.zeros(6)], dim=0)
        pose_hand = rotvec[22:52, :3].contiguous().view(-1)
        betas = shape.view(1, 10)

        data_dict = {
            'cam_name': '1',
            'frame_idx': 0,
            'smpl_params': {
                'betas': betas,
                'trans': trans,
                'root_orient': root_orient,
                'pose_body': pose_body,
                'pose_hand': pose_hand
            }
        }
        # 检查队列大小，如果超过最大容量，则丢弃最旧的帧
        if queue.qsize() >= 30:
            while not queue.empty():

                queue.get()  # 丢弃最旧的帧

        
        queue.put(data_dict)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    # 预测序列，具体参考readme.md文件
    config.dataset.predict_seq = 1


    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    config.mode = 'view'
    fix_random(config.seed)
    app = QApplication(sys.argv)
    mainWindow = MainWindow(config.dataset.img_hw)
    mainWindow.show()

    t1 = threading.Thread(target=renderimg, args=(config, None, mainWindow.get_angles, mainWindow.smplQueue, mainWindow.renderQueue))
    t1.start()
    # t1.join()
    time.sleep(0.001)
    t2 = threading.Thread(target=gsrender, args=(mainWindow.renderQueue, mainWindow.imgQueue))
    t2.start()
    # t1.join()
    time.sleep(0.001)
    t3 = threading.Thread(target=getsmpl_net, args=('', 8082, mainWindow.smplQueue))
    t3.daemon = True
    t3.start()
    time.sleep(0.001)

    mainWindow.update_image()
    
    sys.exit(app.exec_())



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()



