from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import sys

import torch

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cv2
from tqdm import trange
from gaussian_rendererforview import render
from utils.general_utils import fix_random
import hydra
from omegaconf import OmegaConf

from datetime import datetime
import time
import threading
from utils.graphics_utils import getWorld2View2, getProjectionMatrix 

# 定义队列类
class Queue:
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.insert(0, item)

    def pop(self):
        if not self.is_empty():
            return self.queue.pop()
        else:
            return None

    def peek(self):
        if not self.is_empty():
            return self.queue[-1]
        else:
            return None

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.times = 0
        self.render_timer = QTimer(self)
        self.last_x, self.last_y = None, None
        self.horizontal_angle = 0  # 定义水平角度
        self.vertical_angle = 0  # 定义垂直角度
        self.imgQueue = Queue()  # 实例化图像队列

    def initUI(self):
        # 状态标签
        self.statusLabel = QLabel('times: 0', self)
        self.statusLabel.setGeometry(10, 10, 200, 30)
        # 图像显示标签
        self.imageLabel = QLabel(self)
        self.imageLabel.setGeometry(275, 50, 850, 1400)
        self.setFixedSize(1400, 1430)
        self.setWindowTitle('3DGS Rendering with PyQt5')

        self.leftButton = QPushButton('Left', self)
        self.leftButton.setGeometry(250, 10, 200, 30)
        self.leftButton.clicked.connect(self.move_camera_left)
        
        self.rightButton = QPushButton('Right', self)
        self.rightButton.setGeometry(500, 10, 200, 30)
        self.rightButton.clicked.connect(self.move_camera_right)

        self.upButton = QPushButton('Up', self)
        self.upButton.setGeometry(750, 10, 200, 30)
        self.upButton.clicked.connect(self.move_camera_up)

        self.downButton = QPushButton('Down', self)
        self.downButton.setGeometry(1000, 10, 200, 30)
        self.downButton.clicked.connect(self.move_camera_down)

    def update_image(self): 
        # 开启定时器以更新图像
        self.render_timer.start(20)
        self.render_timer.timeout.connect(self.update_frame)

    def update_frame(self):
        # 更新图像帧
        if not self.imgQueue.is_empty():
            now = datetime.now()
            self.times += 1
            print(f"Updating frame:{self.times}", now)
            self.image = self.imgQueue.pop()
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            q_image = QImage(self.image.astype(np.uint8).data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            q_imagescaled = q_image.scaled(850, 1400, Qt.AspectRatioMode.KeepAspectRatio)
            pixmap = QPixmap.fromImage(q_imagescaled)
            
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.repaint() 

            self.statusLabel.setText(f'times: {self.times}')
        self.render_timer.start(20)

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

    def mouseReleaseEvent(self, event):
        # 鼠标释放时重置位置
        if event.button() == Qt.LeftButton:
            self.last_x, self.last_y = None, None

    def rotate_camera(self, dx, dy):
        self.horizontal_angle += dx / 10.0  # 根据需要调整灵敏度
        self.vertical_angle += dy / 10.0  # 根据需要调整灵敏度
        self.horizontal_angle = self.horizontal_angle % 360  # 保持角度在360度内
        self.vertical_angle = max(min(self.vertical_angle, 90), -90)  # 限制垂直角度在-90到90度之间

    def get_angles(self):
        return self.horizontal_angle, self.vertical_angle

# 移动相机函数
def move_camera(R, T, center_point, horizontal_angle, vertical_angle):
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

    inv_E = np.linalg.inv(Ei)
    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if center_point is not None:
        campos -= center_point

    rot_campos = rotation_matrix @ campos
    rot_camrot = rotation_matrix @ camrot  
    if center_point is not None:
        rot_campos += center_point

    return (rot_camrot.T, -rot_camrot.T.dot(rot_campos))


def renderimg(config, scene, imgQueue, get_angles):
    from scene import Scene
    from scene import GaussianModel
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
    center_point = np.array([0.004311501048505306, 0.7597708702087402, 3.8142313957214355], np.float32)

    for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
        view = scene.test_dataset[idx]
        # 修改渲染角度
        horizontal_angle, vertical_angle = get_angles()
        if horizontal_angle != 0 or vertical_angle != 0:
            view.R, view.T = move_camera(view.R, view.T, center_point, -horizontal_angle, vertical_angle)

            view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T,1.0)).transpose(0, 1).cuda()
            view.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100., fovX=view.FoVx,
                                                        fovY=view.FoVy).transpose(0, 1).cuda()
            view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.data['camera_center'] = view.world_view_transform.inverse()[3, :3]
        # 渲染模块
        print(view.R, view.T)
        rendering = render(view, config.opt.iterations, scene, config.pipeline, background,
                            compute_loss=False, return_opacity=False)

        imgQueue.push(np.transpose(rendering.cpu().detach().numpy(), (1, 2, 0)) * 255)
        if imgQueue.size() > 30:
            time.sleep(imgQueue.size() / 30)
        print(imgQueue.size())
        time.sleep(0.0001)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.mode = 'view'
    config.dataset.preload = False
    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    fix_random(config.seed)

    # 创建一个QApplication实例
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()

    # 创建并启动一个新线程用于渲染图像
    t1 = threading.Thread(target=renderimg, args=(config, None, mainWindow.imgQueue, mainWindow.get_angles))
    t1.start()
    time.sleep(1)
    # 更新图像
    mainWindow.update_image()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
