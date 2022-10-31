import numpy as np
import cv2
import torch
import torch.nn as nn
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
n_light = 50
plane_height = 10
size = 1500
n_rays = int(size*size*0.05)
out_path = r'out/test_run/test3/'
code_path = './'
code_backup(out_path, code_path)
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
imgs_path = r'test_data\textures\texture_only\metal_rusty_tj3rdiobw\render_imgs/'
# pc = torch.from_numpy(np.loadtxt(r'test_data\fliter_MVS_pc.txt').astype(np.float32))[:,:3]

texture_gd = get_texture(r'test_data\textures\texture_only\metal_rusty_tj3rdiobw\tex/')[H//2-size//2:H//2-size//2+size, W//2-size//2:W//2-size//2+size]

camera = Camera(size,size,k,W2C,mask)
normal_0 = camera.get_N_from_depth_dif(torch.ones(size,size)*plane_height)
# # cv2.imwrite(out_path+'normal111.png', (normal_0.detach().numpy()*255).astype(np.uint8))
# camera.write_N_img(normal_0, out_path=out_path)
ggxrender = GGXRenderer(use_cuda=False)


imgs_tensor, light_idx = read_imgs_asTensor(imgs_path, n_light, size)
print(imgs_tensor.shape)
# cv2.imwrite('out/i.png', (torch.mean(imgs_tensor, dim=0).numpy()*255).astype(np.uint8))
light_p = torch.from_numpy(generate_light_p(7, 3.2, 1000))[light_idx]     # 世界坐标系下的

new_mask = camera.new_mask
albedo_ref = 0.3*torch.mean(imgs_tensor, dim=0)
depth_ref = torch.ones_like(albedo_ref.clone()[:,:,0])*plane_height

# albedo_0 = nn.Parameter(albedo_ref.clone())
gamma = nn.Parameter(torch.tensor(0.4))
depth_0_delta = nn.Parameter(torch.ones_like(depth_ref))
roughness_0 = texture_gd[:,:,6:9]#torch.ones_like(albedo_ref.clone())*0.5
specular_0 = texture_gd[:,:,9:]#torch.ones_like(albedo_ref.clone())*0.2


##############################################################################
##############################################################################
opt = torch.optim.Adam([depth_0_delta, gamma], lr = 1e-4)
Loss = nn.MSELoss()
L1Loss = nn.L1Loss()

# total_perceptual_loss = VGGLoss()
# perceptual_loss134 = Perceptual_loss134()

alpha = 10
beta = 10

gif_img = []
loss_list = []

rays_idx = torch.where(torch.rand(size,size)<100)
rays_idx_ = torch.cat((rays_idx[0][:,None], rays_idx[1][:,None]), dim=-1)

for i in tqdm.tqdm(range(100)):
    opt.zero_grad()

    depth__ = alpha*depth_0_delta
    imgXYZ = camera.get_imgXYZ_from_rays(depth__[rays_idx_[:,0], rays_idx_[:,1]][:, None], rays_idx_)
    normal_0 = camera.get_N_from_depth_dif(depth__)

    img_l_d = light_p[:,None,:] - imgXYZ

    textures = torch.cat([(normal_0+1)/2, torch.pow(albedo_ref, 1+gamma), roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
    # print(camera.vs_d.shape, img_l_d.shape, textures.shape, '-------------------')
    result = render_pixels(camera.vs_d[rays_idx_[:,0],rays_idx_[:,1],:], ggxrender, img_l_d, textures, 300).reshape(n_light, size, size, 3)


    diff_img = torch.zeros_like(imgs_tensor)
    diff_img[:,camera.new_mask>0,:] = result[:,camera.new_mask>0] - torch.pow(imgs_tensor[:,camera.new_mask>0], 1/2.2)
    diff_img = torch.sum(torch.sum(torch.abs(diff_img), dim=0), dim=-1)[:,:,None].repeat(1,1,3)
    diff_img = diff_img / torch.max(diff_img)

    # if i > 1:
    #     rays_idx_ = select_rays(diff_img[:,:,0], n_rays)

    if i>500:
        bad_rays_idx = select_rays(diff_img[:,:,0], n_rays)
    else:
        bad_rays_idx = rays_idx_
    # loss_img = Loss(torch.pow(result[:,camera.new_mask>0], 2.2), imgs_tensor[:,camera.new_mask>0])
    # loss_img = L1Loss(torch.pow(result[:,camera.new_mask>0], diff_img[None,camera.new_mask>0]), torch.pow(torch.pow(imgs_tensor[:,camera.new_mask>0], diff_img[None,camera.new_mask>0]), 1/2.2))#torch.sum(torch.abs(torch.pow(result[:,camera.new_mask>0], 2.2)-imgs_tensor[:,camera.new_mask>0]))
    # loss_img = Loss(result[:,camera.new_mask>0], torch.pow(imgs_tensor[:,camera.new_mask>0], 1/2.2))#torch.sum(torch.abs(torch.pow(result[:,camera.new_mask>0], 2.2)-imgs_tensor[:,camera.new_mask>0]))
    loss_img = Loss(result[:,bad_rays_idx[:,0], bad_rays_idx[:,1]], torch.pow(imgs_tensor[:,bad_rays_idx[:,0], bad_rays_idx[:,1]], 1/2.2))#torch.sum(torch.abs(torch.pow(result[:,camera.new_mask>0], 2.2)-imgs_tensor[:,camera.new_mask>0]))

    # loss_albedo = Loss(albedo_0[camera.new_mask>0], albedo_ref[camera.new_mask>0])
    loss_depth = Loss(depth__[camera.new_mask>0], depth_ref[camera.new_mask>0])
    #loss_smooth = smooth_loss(depth_0_delta, camera.new_mask)#torch.sum(torch.abs(kornia.filters.laplacian(depth_0_delta[None, None, :, :], 3)[0,0,camera.new_mask>0]))
    loss = loss_img #+ 10*loss_depth #+ 0.1*(1-gamma)**2#+ 1e-3*loss_smooth #+ 0.01*loss_per#+ 1e*loss_depth #+ 0*loss_smooth
    loss.backward()
    if i>-1:
        depth_0_delta.grad = depth_0_delta.grad*diff_img[:,:,0]
    grad_img = torch.abs(depth_0_delta.grad)[:,:,None].repeat(1,1,3)
    grad_img = grad_img/torch.max(grad_img)
    opt.step()
    print('\n', i, loss.item(), loss_depth.item(), gamma.item(), torch.max(torch.abs(depth_0_delta.grad)))

    # vis result
    loss_list.append(torch.log10(loss).item())

    n_img = ((normal_0.detach().numpy()+1)/2)[:,:,[2,1,0]]
    n_img[camera.new_mask.numpy()<1]=0
    # row1 = np.hstack((torch.pow(imgs_tensor[0], 1/2.2).numpy(), texture_gd[:,:,3:6].numpy(), texture_gd[:,:,:3].numpy()[:,:,[2,1,0]]))
    row1 = np.hstack((torch.pow(imgs_tensor[0], 1/2.2).numpy(), grad_img.detach().numpy(), texture_gd[:,:,:3].numpy()[:,:,[2,1,0]]))
    # row2 = np.hstack((result[0].detach().numpy(), torch.pow(albedo_ref, 1+gamma).detach().numpy(), n_img))
    row2 = np.hstack((result[0].detach().numpy(), diff_img.detach().numpy(), n_img))

    cv2.imwrite(out_path+'diff.png', (diff_img.detach().numpy()*255).astype(np.uint8))

    cv2.imwrite(out_path+'render_res.png', (np.vstack((row1, row2))*255).astype(np.uint8))
    cv2.imwrite(out_path+'n_img.png'.format(i), (n_img*255).astype(np.uint8))
    # np.save(out_path+'depth', (alpha*depth_0_delta).detach().numpy())
    gif_img.append((n_img*255).astype(np.uint8)[:,:,[2,1,0]])
    plt.plot(loss_list)
    plt.savefig(out_path+'loss.png')
    plt.close()
imageio.mimsave(out_path+'normal.gif', gif_img, fps=10)
# vis_depth_with_color((depth__).detach().numpy(), torch.pow(result[0], 2.2).detach().numpy(), camera.new_mask)


##################################
opt = torch.optim.SGD([depth_0_delta], lr = 1e-3)
Loss = nn.MSELoss()
L1Loss = nn.L1Loss()

# total_perceptual_loss = VGGLoss()
# perceptual_loss134 = Perceptual_loss134()

alpha = 10
beta = 10

gif_img = []
loss_list = []

rays_idx = torch.where(torch.rand(size,size)<100)
rays_idx_ = torch.cat((rays_idx[0][:,None], rays_idx[1][:,None]), dim=-1)

for i in tqdm.tqdm(range(100)):
    opt.zero_grad()

    depth__ = alpha*depth_0_delta
    imgXYZ = camera.get_imgXYZ_from_rays(depth__[rays_idx_[:,0], rays_idx_[:,1]][:, None], rays_idx_)
    normal_0 = camera.get_N_from_depth_dif(depth__)

    img_l_d = light_p[:,None,:] - imgXYZ

    textures = torch.cat([(normal_0+1)/2, torch.pow(albedo_ref, 1+gamma), roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
    # print(camera.vs_d.shape, img_l_d.shape, textures.shape, '-------------------')
    result = render_pixels(camera.vs_d[rays_idx_[:,0],rays_idx_[:,1],:], ggxrender, img_l_d, textures, 300).reshape(n_light, size, size, 3)


    diff_img = torch.zeros_like(imgs_tensor)
    diff_img[:,camera.new_mask>0,:] = result[:,camera.new_mask>0] - torch.pow(imgs_tensor[:,camera.new_mask>0], 1/2.2)
    diff_img = torch.sum(torch.sum(torch.abs(diff_img), dim=0), dim=-1)[:,:,None].repeat(1,1,3)
    diff_img = diff_img / torch.max(diff_img)

    # if i > 1:
    #     rays_idx_ = select_rays(diff_img[:,:,0], n_rays)

    if i>500:
        bad_rays_idx = select_rays(diff_img[:,:,0], n_rays)
    else:
        bad_rays_idx = rays_idx_
    # loss_img = Loss(torch.pow(result[:,camera.new_mask>0], 2.2), imgs_tensor[:,camera.new_mask>0])
    # loss_img = L1Loss(torch.pow(result[:,camera.new_mask>0], diff_img[None,camera.new_mask>0]), torch.pow(torch.pow(imgs_tensor[:,camera.new_mask>0], diff_img[None,camera.new_mask>0]), 1/2.2))#torch.sum(torch.abs(torch.pow(result[:,camera.new_mask>0], 2.2)-imgs_tensor[:,camera.new_mask>0]))
    # loss_img = Loss(result[:,camera.new_mask>0], torch.pow(imgs_tensor[:,camera.new_mask>0], 1/2.2))#torch.sum(torch.abs(torch.pow(result[:,camera.new_mask>0], 2.2)-imgs_tensor[:,camera.new_mask>0]))
    loss_img = Loss(result[:,bad_rays_idx[:,0], bad_rays_idx[:,1]], torch.pow(imgs_tensor[:,bad_rays_idx[:,0], bad_rays_idx[:,1]], 1/2.2))#torch.sum(torch.abs(torch.pow(result[:,camera.new_mask>0], 2.2)-imgs_tensor[:,camera.new_mask>0]))

    # loss_albedo = Loss(albedo_0[camera.new_mask>0], albedo_ref[camera.new_mask>0])
    loss_depth = Loss(depth__[camera.new_mask>0], depth_ref[camera.new_mask>0])
    #loss_smooth = smooth_loss(depth_0_delta, camera.new_mask)#torch.sum(torch.abs(kornia.filters.laplacian(depth_0_delta[None, None, :, :], 3)[0,0,camera.new_mask>0]))
    loss = loss_img #+ 10*loss_depth #+ 0.1*(1-gamma)**2#+ 1e-3*loss_smooth #+ 0.01*loss_per#+ 1e*loss_depth #+ 0*loss_smooth
    loss.backward()
    if i>-1:
        depth_0_delta.grad = depth_0_delta.grad*diff_img[:,:,0]
    grad_img = torch.abs(depth_0_delta.grad)[:,:,None].repeat(1,1,3)
    grad_img = grad_img/torch.max(grad_img)
    opt.step()
    print('\n', i, loss.item(), loss_depth.item(), gamma.item(), torch.max(torch.abs(depth_0_delta.grad)))

    # vis result
    loss_list.append(torch.log10(loss).item())

    n_img = ((normal_0.detach().numpy()+1)/2)[:,:,[2,1,0]]
    n_img[camera.new_mask.numpy()<1]=0
    # row1 = np.hstack((torch.pow(imgs_tensor[0], 1/2.2).numpy(), texture_gd[:,:,3:6].numpy(), texture_gd[:,:,:3].numpy()[:,:,[2,1,0]]))
    row1 = np.hstack((torch.pow(imgs_tensor[0], 1/2.2).numpy(), grad_img.detach().numpy(), texture_gd[:,:,:3].numpy()[:,:,[2,1,0]]))
    # row2 = np.hstack((result[0].detach().numpy(), torch.pow(albedo_ref, 1+gamma).detach().numpy(), n_img))
    row2 = np.hstack((result[0].detach().numpy(), diff_img.detach().numpy(), n_img))

    cv2.imwrite(out_path+'diff.png', (diff_img.detach().numpy()*255).astype(np.uint8))

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