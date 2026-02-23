# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import time
import argparse
import numpy as np

import cv2
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt

plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')

# UI and OpenGL
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm

from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth
from PIL import Image
from matplotlib import cm
# Argument Parser
parser = argparse.ArgumentParser(description='NeWCRFs Live 3D')
parser.add_argument('--model_name',      type=str,   help='model name', default='newcrfs')
parser.add_argument('--encoder',         type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--max_depth',       type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str,   help='path to a checkpoint to load',default='model_nyu.ckpt')
parser.add_argument('--input_height',    type=int,   help='input height', default=480)
parser.add_argument('--input_width',     type=int,   help='input width',  default=640)
parser.add_argument('--dataset',         type=str,   help='dataset this model trained on',  default='nyu')
parser.add_argument('--crop',            type=str,   help='crop: kbcrop, edge, non',  default='non')
parser.add_argument('--video',           type=str,   help='video path',  default='../seq_02.mp4')

args = parser.parse_args()

# Image shapes
height_rgb, width_rgb = args.input_height, args.input_width
height_depth, width_depth = height_rgb, width_rgb


# =============== Intrinsics rectify ==================
# Open this if you have the real intrinsics
Use_intrs_remap = False
# Intrinsic parameters for your own webcam camera
camera_matrix = np.zeros(shape=(3, 3))
camera_matrix[0, 0] = 5.4765313594010649e+02
camera_matrix[0, 2] = 3.2516069906172453e+02
camera_matrix[1, 1] = 5.4801781476172562e+02
camera_matrix[1, 2] = 2.4794113960783835e+02
camera_matrix[2, 2] = 1
dist_coeffs = np.array([ 3.7230261423972011e-02, -1.6171708069773008e-01, -3.5260752900266357e-04, 1.7161234226767313e-04, 1.0192711400840315e-01 ])
# Parameters for a model trained on NYU Depth V2
new_camera_matrix = np.zeros(shape=(3, 3))
new_camera_matrix[0, 0] = 518.8579
new_camera_matrix[0, 2] = 320
new_camera_matrix[1, 1] = 518.8579
new_camera_matrix[1, 2] = 240
new_camera_matrix[2, 2] = 1

R = np.identity(3, dtype=np.float)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R, new_camera_matrix, (width_rgb, height_rgb), cv2.CV_32FC1)

# =============Functions=====
def load_model():
    args.mode = 'test'
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    return model

# Function timing
ticTime = time.time()


def tic():
    global ticTime;
    ticTime = time.time()


def toc():
    print('{0} seconds.'.format(time.time() - ticTime))


# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888).copy()


def qimage_to_np(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return np.array(img.constBits()).reshape(img.height(), img.width(), 4)


# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


def loadModel():
        print('== loadModel')
        tic()
        model = load_model()
        print('Model loaded.')
        toc()

model = load_model()

#======================
tic()

im = cv2.imread("test.jpg")


frame_ud = cv2.resize(im, (width_rgb, height_rgb), interpolation=cv2.INTER_LINEAR)
frame = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2RGB)
image = np_to_qimage(frame)
input_image = frame[:, :, :3].astype(np.float32)

# Normalize image
input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017

H, W, _ = input_image.shape
if args.crop == 'kbcrop':
    top_margin = int(H - 352)
    left_margin = int((W - 1216) / 2)
    input_image_cropped = input_image[top_margin:top_margin + 352, 
                                      left_margin:left_margin + 1216]
elif args.crop == 'edge':
    input_image_cropped = input_image[32:-32, 32:-32, :]
else:
    input_image_cropped = input_image

input_images = np.expand_dims(input_image_cropped, axis=0)
input_images = np.transpose(input_images, (0, 3, 1, 2))

with torch.no_grad():
    image = Variable(torch.from_numpy(input_images)).cuda()
    # Predict
    depth_est = model(image)
    post_process = True
    if post_process:
        image_flipped = flip_lr(image)
        depth_est_flipped = model(image_flipped)
        depth_cropped = post_process_depth(depth_est, depth_est_flipped)

depth = np.zeros((height_depth, width_depth), dtype=np.float32)


if args.crop == 'kbcrop':
    depth[top_margin:top_margin + 352, left_margin:left_margin + 1216] = \
            depth_cropped[0].cpu().squeeze() / args.max_depth
elif args.crop == 'edge':
    depth[32:-32, 32:-32] = depth_cropped[0].cpu().squeeze() / args.max_depth
else:
    depth[:, :] = depth_cropped[0].cpu().squeeze() / args.max_depth

greysDepth = (greys(np.log10(depth * args.max_depth))[:, :, :3] * 255).astype('uint8')

os.remove('depthcrfs.txt')
f = cv2.FileStorage('depthcrfs.txt',cv2.FILE_STORAGE_WRITE)
f.write('mat1',depth)
f.release()
#np.savetxt('depth.txt',depth,fmt='%.2f',delimiter=',')
#im = Image.fromarray(np.uint8(cm.hot(depth)*255))
#plt.imshow(im)
#plt.show()

print(toc())
