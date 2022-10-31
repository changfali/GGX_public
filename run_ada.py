import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR  # example
from warmup_scheduler_pytorch import WarmUpScheduler
import skimage.io as io
import os
import matplotlib.pyplot as plt
import tqdm
import kornia
import imageio
import time

from GGXRenderer import GGXRenderer
from camera import Camera
from utils import *
# from perceptual_loss import *

t1 = time.time()
H = 2400; W=3200
n_light = 64
light_int = 180
plane_height = 10
size = 1400
n_rays = int(size*size*0.02)
out_path = r'out/test3/wood_board/'
code_path = './'
code_backup(out_path, code_path)
n_epoch = 200
lr = 3e-4
warmup_steps = 5
alpha = 1

k = torch.tensor([[3602, 0.,   1600, 0.],
                  [0.,   3602, 1200, 0.],
                  [0.,   0.,   1., 0.],
                  [0.,   0.,   0., 1.]])
W2C = torch.tensor([[1., 0.,  0., 0.],
                    [0., -1., 0., 0.],
                    [0., 0.,  -1., 10],
                    [0., 0.,  0.,  1]])

k[0,2] -= W//2-size//2
k[1,2] -= H//2-size//2
mask = cv2.imread(r'test_data\mask_full.png')[H//2-size//2:H//2-size//2+size, W//2-size//2:W//2-size//2+size, 0]/255
imgs_path = r'test_data\textures\texture_only\wood_board_wchncbqs\renders2/'
texture_gd = get_texture(r'test_data\textures\texture_only\wood_board_wchncbqs\tex/')[H//2-size//2:H//2-size//2+size, W//2-size//2:W//2-size//2+size]
ggxrender = GGXRenderer(use_cuda=False)
imgs_tensor, light_idx = read_imgs_asTensor(imgs_path, n_light, size)
light_p = torch.from_numpy(generate_light_p(7, 30, 300))[light_idx]     # 世界坐标系下的
# light_p = torch.from_numpy(generate_light_p(7, 3.2, 1000))[light_idx]     # 世界坐标系下的

albedo_ref = 5*torch.mean(imgs_tensor, dim=0)
# albedo_ref = 5*torch.mean(imgs_tensor, dim=0)
gamma = nn.Parameter(torch.tensor(0.))

ratio_list = [8,4,2,1]
epoch_list = [15,6,6,6]
Loss = nn.MSELoss()
L1Loss = nn.L1Loss()
upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

n_ = 0
for ratio in ratio_list:
    n_ += 1
    k_1 = k.clone()
    k_1[:2,:] *= 1./ratio
    mask_1 = mask[::ratio, ::ratio]
    imgs_tensor_1 = imgs_tensor[:,::ratio,::ratio]
    albedo_ref_1 = albedo_ref[::ratio, ::ratio]

    size_1 = size //ratio
    camera = Camera(size_1, size_1, k_1, W2C, mask_1)

    if n_ ==1:
        depth_base = torch.ones_like(albedo_ref_1.clone()[:,:,0])*plane_height

    depth_0_delta = nn.Parameter(torch.zeros_like(depth_base))
    roughness_0 = texture_gd[::ratio,::ratio,6:9]
    specular_0 = texture_gd[::ratio,::ratio,9:]
    opt_adam = torch.optim.Adam([depth_0_delta], lr = ratio*lr)

    loss_list = []
    rays_idx = torch.where(torch.rand(size_1,size_1)<100)
    rays_idx_ = torch.cat((rays_idx[0][:,None], rays_idx[1][:,None]), dim=-1)

    result_0 = torch.zeros(n_light,size_1, size_1, 3)
    plane_XYZ = camera.get_planeXYZ_fromAdepth(10)
    gif_img = []
    for i in tqdm.tqdm(range(epoch_list[n_-1])):
        opt_adam.zero_grad()

        depth__ = depth_0_delta+depth_base
        imgXYZ = camera.get_imgXYZ_from_rays(depth__[rays_idx_[:,0], rays_idx_[:,1]][:, None], rays_idx_)
        normal_0 = camera.get_N_from_depth_dif(depth__)
        img_l_d = light_p[:,None,:] - plane_XYZ.reshape(-1,3)

        textures = torch.cat([(normal_0+1)/2, torch.pow(albedo_ref_1, 1+gamma), roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
        result = render_pixels(camera.vs_d[rays_idx_[:,0],rays_idx_[:,1],:], ggxrender, img_l_d, textures, light_int)#.reshape(n_light, size, size, 3)

        result_0[:,rays_idx_[:,0],rays_idx_[:,1],:] = torch.pow(result, 1/2.2)
        result_0[:,camera.new_mask<1]=0.
        diff_img = torch.zeros_like(imgs_tensor_1)
        diff_img[:,camera.new_mask>0,:] = result_0[:,camera.new_mask>0] - torch.pow(imgs_tensor_1[:,camera.new_mask>0], 1/2.2)
        diff_img = torch.sum(torch.sum(torch.abs(diff_img), dim=0), dim=-1)[:,:,None].repeat(1,1,3)
        diff_img = diff_img / torch.max(diff_img)
        loss_img = L1Loss(tone_mapping_img(result), tone_mapping_img(imgs_tensor_1[:,rays_idx_[:,0], rays_idx_[:,1]]))

        loss = loss_img#+0*loss_img_fft
        loss.backward()
        grad_img = torch.abs(depth_0_delta.grad)[:,:,None].repeat(1,1,3)
        grad_img = grad_img/torch.max(grad_img)
        print('\n', ratio, i, depth__.shape, loss.item(), gamma.item(), loss_img.item())

        opt_adam.step()

        # vis result
        loss_list.append(torch.log10(loss).item())

        n_img = ((normal_0.detach().numpy()+1)/2)[:,:,[2,1,0]]
        n_img[camera.new_mask.numpy()<1] = 0
        row1 = np.hstack((torch.pow(imgs_tensor_1[0], 1/2.2).numpy(), grad_img.detach().numpy(), texture_gd[::ratio,::ratio,:3].numpy()[:,:,[2,1,0]], texture_gd[::ratio,::ratio,3:6].numpy()))
        row2 = np.hstack((result_0[0].detach().numpy(), diff_img.detach().numpy(), n_img,  torch.pow(albedo_ref_1, 1+gamma).detach().numpy()))

        cv2.imwrite(out_path+'diff.png', (diff_img.detach().numpy()*255).astype(np.uint8))
        cv2.imwrite(out_path+'grad.png', (grad_img.detach().numpy()*255).astype(np.uint8))

        cv2.imwrite(out_path+'render_res.png', (np.vstack((row1, row2))*255).astype(np.uint8))
        cv2.imwrite(out_path+'n_img.png', (n_img*255).astype(np.uint8))

        plt.plot(loss_list)
        plt.savefig(out_path+'loss.png')
        plt.close()
        gif_img.append((np.vstack((row1, row2))*255).astype(np.uint8)[:,:,[2,1,0]])
    depth_base = upsample(depth__.detach()[None, None, :])[0,0]
    imageio.mimsave(out_path+'render_res_{}.gif'.format(ratio), gif_img, fps=3)

# imageio.mimsave(out_path+'normal.gif', gif_img, fps=10)
np.save(out_path+'render_img.npy', torch.pow(result_0[0], 2.2).detach().numpy())
np.save(out_path+'depth.npy', (depth__).detach().numpy())
t2 = time.time()
print('Time:', (t2-t1)/60)
vis_depth_with_color((depth__).detach().numpy(), torch.pow(result_0[0], 1).detach().numpy(), camera.new_mask)