import numpy as np
import cv2
import torch
import torch.nn as nn
import skimage.io as io
import os

from GGXRenderer import GGXRenderer
from camera import Camera
from utils import *


k = torch.tensor([[3602, 0.,   1600, 0.],
                  [0.,   3602, 1200, 0.],
                  [0.,   0.,   1., 0.],
                  [0.,   0.,   0., 1.]])
W2C = torch.tensor([[1., 0.,  0., 0.],
                    [0., -1., 0., 0.],
                    [0., 0.,  -1., 10],
                    [0., 0.,  0.,  1]])


size = 3200
hight = 7
r = 30
n_light = 300
light_int = 180

# imgs_path = r'D:\changfa\PBR\Textures\quixel_20200322\Brick_Modern_udqfejrew_8K_surface_ms/'
imgs_path = r'D:\changfa\PBR\Textures\quixel_20200322\Marble_Polished_udnjdf2n_8K_surface_ms/'

outpath = imgs_path + 'renders_{}_{}_{}_{}/'.format(hight, r, n_light, light_int)
if not os.path.exists(outpath):
    os.makedirs(outpath)

textures = get_texture_v2(imgs_path)[:2400,:3200]
print('textures:', textures.shape)
mask = cv2.imread(r'test_data\mask.png')[:,:,0]/255
# pc = torch.from_numpy(np.loadtxt(r'test_data\fliter_MVS_pc.txt').astype(np.float32))[:,:3]

camera = Camera(3200,2400,k,W2C,mask)
ggxrender = GGXRenderer(use_cuda=False)

###################################################################
plane_XYZ = camera.get_planeXYZ_fromAdepth(10)   # 世界坐标系下

# light_p = torch.tensor([[0,0,10]]).repeat(n_light,1)# 世界坐标系下
light_p = torch.from_numpy(generate_light_p(hight, r, n_light))

N_light_batch = 30
light_p_split = torch.split(light_p, N_light_batch, 0)
for j, lp in enumerate(light_p_split):
    img_l_d = lp[:,None, None,:] - plane_XYZ
    result = render_img(camera.vs_d, ggxrender, img_l_d, textures, light_int)
    print(result.shape)
    for i in range(result.shape[0]):
        print(j, i)
        io.imsave(outpath + 'img_{}.png'.format(int(j*N_light_batch+i)), ((result[i].reshape(camera.H,camera.W,3).numpy())*255).astype(np.uint8)[:,:,[2,1,0]])

