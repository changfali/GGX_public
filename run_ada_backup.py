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

from GGXRenderer import GGXRenderer
from camera import Camera
from utils import *
# from perceptual_loss import *

H = 2400; W=3200
n_light = 64
light_int = 180
plane_height = 10
size = 1200
n_rays = int(size*size*0.05)
out_path = r'out/test_run/test17/'
code_path = './'
code_backup(out_path, code_path)
n_epoch = 60
lr = 1e-3
warmup_steps = 5
alpha = 1
# beta = 10

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
imgs_path = r'test_data\textures\texture_only\brick_modern_vizcegj\renders/'
texture_gd = get_texture(r'test_data\textures\texture_only\brick_modern_vizcegj\tex/')[H//2-size//2:H//2-size//2+size, W//2-size//2:W//2-size//2+size]

camera = Camera(size,size,k,W2C,mask)
normal_0 = camera.get_N_from_depth_dif(torch.ones(size,size)*plane_height)
ggxrender = GGXRenderer(use_cuda=False)
imgs_tensor, light_idx = read_imgs_asTensor(imgs_path, n_light, size)
light_p = torch.from_numpy(generate_light_p(7, 3.2, 1000))[light_idx]     # 世界坐标系下的

new_mask = camera.new_mask
albedo_ref = 0.3*torch.mean(imgs_tensor, dim=0)
depth_ref = torch.ones_like(albedo_ref.clone()[:,:,0])*plane_height
gamma = nn.Parameter(torch.tensor(0.))
depth_0_delta = nn.Parameter(torch.zeros_like(depth_ref))
roughness_0 = texture_gd[:,:,6:9]
specular_0 = texture_gd[:,:,9:]


##############################################################################
##############################################################################
opt_gamma = torch.optim.Adam([gamma], lr = 10*lr)
opt_adam = torch.optim.Adam([depth_0_delta], lr = 0.1*lr)
opt_SGD = torch.optim.SGD([depth_0_delta], lr = 500*lr)

# lr_scheduler = CosineAnnealingLR(opt, T_max=1e9)
# warmup_scheduler = WarmUpScheduler(opt, lr_scheduler,
#                                 #    len_loader=n_epoch,
#                                    warmup_steps=warmup_steps,
#                                    warmup_start_lr=1e-7,
#                                    warmup_mode='linear')
Loss = nn.MSELoss()
L1Loss = nn.L1Loss()

gif_img = []
loss_list = []

rays_idx = torch.where(torch.rand(size,size)<100)
rays_idx_ = torch.cat((rays_idx[0][:,None], rays_idx[1][:,None]), dim=-1)

depth_base = 10*torch.ones_like(depth_ref)

for i in tqdm.tqdm(range(n_epoch)):
    if i<10:
        opt_gamma.zero_grad()
    elif i <40 :
        opt_adam.zero_grad()
    else:
        opt_SGD.zero_grad()
    depth__ = 3*depth_0_delta+depth_base
    imgXYZ = camera.get_imgXYZ_from_rays(depth__[rays_idx_[:,0], rays_idx_[:,1]][:, None], rays_idx_)
    normal_0 = camera.get_N_from_depth_dif(depth__)
    img_l_d = light_p[:,None,:] - imgXYZ

    textures = torch.cat([(normal_0+1)/2, torch.pow(albedo_ref, 1+gamma), roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
    # textures = torch.cat([(normal_0+1)/2, texture_gd[:,:,3:6], roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
    result = render_pixels(camera.vs_d[rays_idx_[:,0],rays_idx_[:,1],:], ggxrender, img_l_d, textures, light_int).reshape(n_light, size, size, 3)

    diff_img = torch.zeros_like(imgs_tensor)
    diff_img[:,camera.new_mask>0,:] = result[:,camera.new_mask>0] - torch.pow(imgs_tensor[:,camera.new_mask>0], 1/2.2)
    diff_img = torch.sum(torch.sum(torch.abs(diff_img), dim=0), dim=-1)[:,:,None].repeat(1,1,3)
    diff_img = diff_img / torch.max(diff_img)

    bad_rays_idx = rays_idx_
    loss_img = Loss(result[:,bad_rays_idx[:,0], bad_rays_idx[:,1]], torch.pow(imgs_tensor[:,bad_rays_idx[:,0], bad_rays_idx[:,1]], 1/2.2))
    loss = loss_img
    loss.backward()
    grad_img = torch.abs(depth_0_delta.grad)[:,:,None].repeat(1,1,3)
    grad_img = grad_img/torch.max(grad_img)
    print('\n', i, loss.item(), gamma.item())
    if i<10:
        opt_gamma.step()
    elif i<50 :
        opt_adam.step()
    else:
        depth_0_delta.grad = (torch.pow(diff_img[:,:,0], 1/3.2)*depth_0_delta.grad)
        opt_SGD.step()
    # warmup_scheduler.step()


    # vis result
    loss_list.append(torch.log10(loss).item())

    n_img = ((normal_0.detach().numpy()+1)/2)[:,:,[2,1,0]]
    n_img[camera.new_mask.numpy()<1] = 0
    # row1 = np.hstack((torch.pow(imgs_tensor[0], 1/2.2).numpy(), texture_gd[:,:,3:6].numpy(), texture_gd[:,:,:3].numpy()[:,:,[2,1,0]]))
    row1 = np.hstack((torch.pow(imgs_tensor[0], 1/2.2).numpy(), grad_img.detach().numpy(), texture_gd[:,:,:3].numpy()[:,:,[2,1,0]], texture_gd[:,:,3:6].numpy()))
    # row2 = np.hstack((result[0].detach().numpy(), torch.pow(albedo_ref, 1+gamma).detach().numpy(), n_img))
    row2 = np.hstack((result[0].detach().numpy(), diff_img.detach().numpy(), n_img,  torch.pow(albedo_ref, 1+gamma).detach().numpy()))

    cv2.imwrite(out_path+'diff.png', (diff_img.detach().numpy()*255).astype(np.uint8))
    cv2.imwrite(out_path+'grad.png', (grad_img.detach().numpy()*255).astype(np.uint8))

    cv2.imwrite(out_path+'render_res.png', (np.vstack((row1, row2))*255).astype(np.uint8))
    cv2.imwrite(out_path+'n_img.png'.format(i), (n_img*255).astype(np.uint8))
    # np.save(out_path+'depth', (alpha*depth_0_delta).detach().numpy())
    gif_img.append((n_img*255).astype(np.uint8)[:,:,[2,1,0]])
    plt.plot(loss_list)
    plt.savefig(out_path+'loss.png')
    plt.close()


imageio.mimsave(out_path+'normal.gif', gif_img, fps=10)
np.save(out_path+'render_img.npy', torch.pow(result[0], 2.2).detach().numpy())
np.save(out_path+'depth.npy', (depth__).detach().numpy())
vis_depth_with_color((depth__).detach().numpy(), torch.pow(result[0], 2.2).detach().numpy(), camera.new_mask)