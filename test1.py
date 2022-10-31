import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import kornia
from utils import *
import torch.nn as nn
import os

# size = 9
# img = torch.from_numpy(cv2.imread(r'out\test_run\test22\n_img.png',0))/255
# # img_blur = cv2.Laplacian(img, cv2.CV_64F)
# img_blur = kornia.filters.laplacian(img[None, None, :, :], 3)
# plt.imshow((img_blur[0,0]**2)**0.3)
# plt.show()
# plt.imshow((img_blur.numpy()))

# img = torch.from_numpy(cv2.imread(r'test_data\textures\texture_only\wood_plank_ujlndfvn\render_imgs\img_1.png')/255)
# # img_fft = torch.abs(torch.fft.fftshift(torch.fft.fftn(img)))#.numpy().transpose(1,2,0)
# # # print(img_fft.shape, img.shape, torch.max(img_fft), torch.min(img_fft), torch.mean(img_fft))
# # plt.imshow(torch.clip(img_fft, max=500).numpy().transpose(1,2,0)[:,:,0])
# # plt.colorbar()
# # plt.show()
# # # vis_depth(img_fft[:,:,1], np.ones_like(img_fft[:,:,1]))
# print(img.shape)
# tone_img = tone_mapping_img(img)
# plt.subplot(211)
# plt.imshow(tone_img)
# plt.subplot(212)
# plt.imshow(img)
# plt.show()
# m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# a = torch.randn(1,1,20,20)
# print(m(a).shape)

imgs_path = r'D:\changfa\PBR\Textures\quixel_20200322\Brick_Modern_udqfejrew_8K_surface_ms/'
imgs_list = os.listdir(imgs_path)

def get_texture(imgs_path):
    Albedo = cv2.imread(imgs_path+'Albedo.jpg')
    Normal = cv2.imread(imgs_path+'Normal.jpg')[:,:,[2,1,0]]
    Roughness = cv2.imread(imgs_path+'Roughness.jpg')
    Specular = cv2.imread(imgs_path+'Specular.jpg')
    tex = torch.tensor(np.concatenate([Normal, Albedo, Roughness, Specular],axis=-1))/255
    tex[:,:,3:6] = torch.pow(tex[:,:,3:6]+ 1e-6, 2.2)
    tex[:,:,-3:] = torch.pow(tex[:,:,-3:]+ 1e-6, 2.2)
    return tex

def get_texture_v2(imgs_path):
    names_list = os.listdir(imgs_path)
    Specular = None
    for name_ in names_list:
        if 'Albedo.jpg' in name_:
            Albedo = cv2.imread(imgs_path+name_)
        elif 'Normal.jpg' in name_:
            Normal = cv2.imread(imgs_path+name_)[:,:,[2,1,0]]
        elif 'Roughness.jpg' in name_:
            Roughness = cv2.imread(imgs_path+name_)
        elif 'Specular.jpg' in name_:
            Specular = cv2.imread(imgs_path+name_)
        else:
            pass
    if Specular is not None:
        tex = torch.tensor(np.concatenate([Normal, Albedo, Roughness, Specular],axis=-1))/255
    else:
        Specular = np.zeros_like(Albedo)
        tex = torch.tensor(np.concatenate([Normal, Albedo, Roughness, Specular],axis=-1))/255
    tex[:,:,3:6] = torch.pow(tex[:,:,3:6]+ 1e-6, 2.2)
    tex[:,:,-3:] = torch.pow(tex[:,:,-3:]+ 1e-6, 2.2)
    return tex

imgs_path = r'D:\changfa\PBR\Textures\quixel_20200322\Brick_Modern_udqfejrew_8K_surface_ms/'
tex = get_texture_v2(imgs_path)
print(tex.shape)
plt.imshow(tex[:,:,9:])
plt.show()