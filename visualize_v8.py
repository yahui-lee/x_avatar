from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import sys

import torch
import numpy as np
import os
from scene import Scene,GaussianModel

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

# torch.cuda.set_device(2)
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.times = 0
        self.render_timer = QTimer(self)
        self.last_x, self.last_y = None, None
        self.horizontal_angle = 0  # 定义水平角度
        self.vertical_angle = 0  # 定义垂直角度
        self.imgQueue = Queue()  

    def initUI(self):
        self.statusLabel = QLabel('times: 0', self)
        self.statusLabel.setGeometry(10, 10, 200, 30)
        self.imageLabel = QLabel(self)
        self.imageLabel.setGeometry(10, 50, 1324, 1324)
        self.setFixedSize(1324, 1354)
        self.setWindowTitle('3DGS Rendering with PyQt5')

        # left button
        self.leftButton = QPushButton('Left', self)
        self.leftButton.setGeometry(250, 10, 200, 30)
        self.leftButton.clicked.connect(self.move_camera_left)
        
        # right button
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
        self.render_timer.start(20)
        self.render_timer.timeout.connect(self.update_frame)
        # if t1.is_alive():
        #     time.sleep(0.0001)


    def update_frame(self):
        if not self.imgQueue.is_empty():
            now = datetime.now()
            self.times += 1
            print(f"Updating frame:{self.times}",now)
            self.image = self.imgQueue.pop()
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            q_image = QImage(self.image.astype(np.uint8).data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            q_imagescaled = q_image.scaled(1324, 1324, Qt.AspectRatioMode.KeepAspectRatio)
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

    def rotate_camera(self, dx, dy):
        self.horizontal_angle += dx / 5.0  # 根据需要调整灵敏度
        self.vertical_angle += dy / 5.0  # 根据需要调整灵敏度
        self.horizontal_angle = self.horizontal_angle % 360  # 保持角度在360度内
        self.vertical_angle = max(min(self.vertical_angle, 90), -90)  # 限制垂直角度在-90到90度之间

    def get_angles(self):
        return self.horizontal_angle, self.vertical_angle

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

def renderimg(config, scene, imgQueue, get_angles):
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

        # scene.test_dataset = list(scene.test_dataset)
        
        for idx in trange(len(scene.test_dataset) , desc="Rendering progress"):
            view = scene.test_dataset[idx % len(scene.test_dataset)]
            horizontal_angle, vertical_angle = get_angles()
            if horizontal_angle != 0 or vertical_angle != 0:
                # update camera parameters
                view.R, view.T = move_camera(view.R, view.T, view.smplth, horizontal_angle, -vertical_angle)
            view.T = view.T.reshape((3, 1))
            # note that in ZJUMoCap the camera center does not align perfectly
            # here we try to offset it by modifying the extrinsic...
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
            view.FoVy = focal2fov(view.K[1, 1], view.image_height) * 1.08
            view.FoVx = focal2fov(view.K[0, 0], view.image_width) * 1.08
            # print( "DIY R:",view.K,view.R,view.T)
            view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, 1.0)).transpose(0, 1).cuda()
            view.projection_matrix  = getProjectionMatrix(znear=0.01, zfar=100, fovX=view.FoVx,
                                                        fovY=view.FoVy).transpose(0, 1).cuda()
            view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.data['camera_center'] = view.world_view_transform.inverse()[3, :3]
            
            print(view.FoVy, view.FoVx,view.data['camera_center'])
            rendering = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            # rendering_np = np.transpose(rendering.cpu().numpy(), (1, 2, 0)) * 255
            # rendering_np = rendering_np.clip(0, 255).astype(np.uint8)

            # # OpenCV 使用 BGR 顺序，如果是 RGB 需要转换
            # if rendering_np.shape[2] == 3:  # 如果是 RGB 图像
            #     rendering_np = cv2.cvtColor(rendering_np, cv2.COLOR_RGB2BGR)

            # cv2.imwrite(f"/home/lyh/pythonproject/xavatar/dance_gs/{idx + 1:06d}.png", rendering_np)
            imgQueue.push(np.transpose(rendering.cpu().numpy(),(1,2,0))*255)
            if imgQueue.size()>40:
                time.sleep(imgQueue.size()/40)
            #print(imgQueue.size())
            time.sleep(0.0001)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    # 预测序列，具体参考readme.md文件
    config.dataset.predict_seq = 2

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    config.mode = 'view'
    fix_random(config.seed)

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    
    t1 = threading.Thread(target=renderimg, args=(config, None, mainWindow.imgQueue, mainWindow.get_angles))
    t1.start()
    # t1.join()
    time.sleep(1)

    mainWindow.update_image()
    
    sys.exit(app.exec_())



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()



