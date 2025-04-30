from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import sys

import torch
import numpy as np
import os

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


    def update_image(self): 
        self.render_timer.start(20)
        self.render_timer.timeout.connect(self.update_frame)
        # if t1.is_alive():
        #     time.sleep(0.0001)


    def update_frame(self):
        if not imgQueue.is_empty():
            now = datetime.now()
            self.times += 1
            print(f"Updating frame:{self.times}",now)
            self.image = imgQueue.pop()
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
        global direction
        direction -= 1

    def move_camera_right(self):
        print("Moving camera right")
        global direction
        direction += 1


def move_camera( R, T, trans, direction, rotate_axis='y', inv_angle=False):
    if direction != 0:
        Ri = np.array(R, np.float32)
        Ti = np.array(T, np.float32)
        Ei = np.eye(4)
        Ei[:3,:3] = Ri
        Ei[:3,3:] = Ti.reshape((3, 1))

        angle = 2 * np.pi * 0.5 * direction


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

        rotate_coord = {'x':0, 'y':1, 'z':2}
        grot_vec = np.array([0., 0., 0.])
        grot_vec[rotate_coord[rotate_axis]] = angle
        grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

        rot_campos = grot_mtx.dot(campos)
        rot_camrot = grot_mtx.dot(camrot)
        if trans is not None:
            rot_campos += trans

        return(rot_camrot.T, -rot_camrot.T.dot(rot_campos))

def renderimg(config, scene):
    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # scene.test_dataset = list(scene.test_dataset)
    
    # for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
    for idx in trange(len(scene.test_dataset), desc="Rendering progress"):

        view = scene.test_dataset[idx]
        if direction != 0:
            view.R, view.T = move_camera(view.R, view.T[:, 0], view.smplth, direction)
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
        # update camera parameters
        view.K[0, :] *= view.image_width / view.original_width
        view.K[1, :] *= view.image_height / view.original_height
        view.FovY = focal2fov(view.K[1, 1], view.image_height)
        view.FovX = focal2fov(view.K[0, 0], view.image_width)
        # print( "DIY R:",view.K,view.R,view.T)
        view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, 1.0)).transpose(0, 1).cuda()
        view.projection_matrix  = getProjectionMatrix(znear=0.01, zfar=100, fovX=view.FovX,
                                                    fovY=view.FovY).transpose(0, 1).cuda()
        view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.data['camera_center'] = view.world_view_transform.inverse()[3, :3]
        
        
        rendering = render(view, config.opt.iterations, scene, config.pipeline, background,
                            compute_loss=False, return_opacity=False)

        imgQueue.push(np.transpose(rendering.cpu().numpy(),(1,2,0))*255)
        if imgQueue.size()>40:
            time.sleep(imgQueue.size()/40)
        print(imgQueue.size())
        time.sleep(0.0001)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    config.mode = 'view'
    fix_random(config.seed)
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

        renderimg(config, scene)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    direction = 0
    imgQueue = Queue()

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()

    t1 = threading.Thread(target=main)
    t1.start()
    # t1.join()
    time.sleep(1)

    mainWindow.update_image()
    
    sys.exit(app.exec_())
