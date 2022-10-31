from torch import fix_
import numpy as np
import cv2
import torch
import torch.nn as nn
import skimage.io as io

from GGXRenderer import GGXRenderer
from camera import Camera
from utils import *
import matplotlib.pyplot as plt
import kornia

# a = torch.randn(256,200,3)
# b = torch.arange(100)
# c = torch.arange(100)+10
# idx = torch.cat((b[:,None],c[:,None]), dim=-1)
# print(idx.shape, a[idx[:,0], idx[:,1]].shape)

# a = torch.ones(40,40)
# a[2,3] = 5; a[20,3] = 5
# print(torch.where(a>1))

##################################

# k = torch.tensor([[3602, 0.,   1600, 0.],
#                 [0.,   3602, 1200, 0.],
#                 [0.,   0.,   1., 0.],
#                 [0.,   0.,   0., 1.]])
# W2C = torch.tensor([[1., 0.,  0., 0.],
#                     [0., -1., 0., 0.],
#                     [0., 0.,  -1., 10],
#                     [0., 0.,  0.,  1]])
# mask = cv2.imread(r'test_data\mask.png')[:,:,0]/255
# camera = Camera(3200,2400,k,W2C,mask)

# # pc = torch.from_numpy(np.loadtxt(r'test_data\fliter_MVS_pc.txt').astype(np.float32))[:,:3]
# # print(torch.max(pc[:,2]), torch.min(pc[:,2]))

# # ######
# # imgxyz, depth = camera.get_imgXYZ_from_depth(pc, pc=True)
# depth = np.load(r'test_data\depth_full000.npy')
# print(depth.shape)
# idx = torch.where(torch.randn(2400,3200)>0.9)
# # idx_ = torch.tensor(idx)
# idx_ = torch.cat((idx[0][:,None], idx[1][:,None]), dim=-1)
# print(idx_.shape)

# depth0 = depth[idx_[:,0], idx_[:,1]][:,None]

# imgxyz_ = camera.get_imgXYZ_from_rays(depth0, idx_)
# print(idx_.shape, depth0.shape, imgxyz_.shape)



########################################################
# diff = cv2.GaussianBlur(cv2.imread(r'out\test_run\diff.png'),(5,5),0)[:,:,0]/255
# diff = cv2.imread(r'out\test_run\diff.png')[:,:,0]/255

# print(diff.shape)
# # vis_depth(diff, np.ones_like(diff))
# plt.imshow(diff)
# plt.show()

# # def select_rays(diff, n_rays):

# prob = (diff/np.sum(diff)).reshape(-1)

# a = np.random.choice(diff.reshape(-1), 1000, replace=False, p = prob)
# print(a.shape)
# h,w = diff.shape
# prob = np.random.rand(h,w)*0.2+0.8
# select_p = prob*diff
# # select_p[select_p<0.3]=0

# # plt.imshow(select_p)
# # plt.colorbar()
# # plt.show()


# def select_rays(diff, n_rays):
#     h,w = diff.shape
#     prob = torch.rand(h,w)*0.2+0.8
#     select_p = prob*diff
#     val, idx = select_p.reshape(-1).topk(k=n_rays)
#     h_idx = idx//h; w_idx = idx%w
#     return torch.cat((h_idx[:,None], w_idx[:,None]), dim=-1)
# idx = select_rays(torch.from_numpy(diff), 1000).numpy()
# print(idx.shape)
# select_p[idx[:,0], idx[:,1]] = 3
# plt.imshow(select_p)
# plt.colorbar()
# plt.show()

# Loss = nn.MSELoss()

# x = nn.Parameter(torch.as_tensor(1.))
# opt = torch.optim.SGD([x], lr = 1e-3)

# for i in range(300):
#     opt.zero_grad()
#     y = x+3
#     loss = (y-10)**2
#     print('初始:', x.item(), x.grad)
#     loss.backward()
#     x.grad = x.grad*1000
#     print('求导:', x.item(), x.grad)
#     opt.step()
#     print('更新:', x.item(), x.grad)
#     print('---')



###############################################
# Loss = nn.MSELoss()
# size = 100
# x = nn.Parameter(torch.rand(size, size))
# opt = torch.optim.SGD([x], lr = 1e-3)
# select_ = torch.where(x<0.1)
# rays_idx_ = torch.cat((select_[0][:,None], select_[1][:,None]), dim=-1)
# # fix = torch.zeros(size,size)
# # fix[rays_idx_] = 1.
# print(rays_idx_)

# for i in range(300):
#     opt.zero_grad()
#     # y = x[10:75,10:75]+3
#     y = x[rays_idx_[:,0], rays_idx_[:,1]]
#     # x[0,0]= 0.0
#     loss = torch.sum((y-10)**2)
#     loss.backward()
#     print(x[0,0],x.grad[0,0], x[10,10], x.grad[10,10], x.grad[rays_idx_[0,0], rays_idx_[0,1]])
#     # x.requires_grad[fix<1] = False
#     # print(i)
#     # x.grad = x.grad*1000
#     opt.step()


#################################################
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
size = 300
n_rays = int(size*size*0.02)
out_path = r'out/test_run/test27/'
code_path = './'
code_backup(out_path, code_path)
n_epoch = 200
lr = 1e-3
warmup_steps = 5
alpha = 1

gamma_step = 70
depth_step = 8000

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
imgs_path = r'test_data\textures\texture_only\wood_plank_ujlndfvn\render_imgs/'
texture_gd = get_texture(r'test_data\textures\texture_only\wood_plank_ujlndfvn\tex/')[H//2-size//2:H//2-size//2+size, W//2-size//2:W//2-size//2+size]

camera = Camera(size,size,k,W2C,mask)
normal_0 = camera.get_N_from_depth_dif(torch.ones(size,size)*plane_height)
ggxrender = GGXRenderer(use_cuda=False)
imgs_tensor, light_idx = read_imgs_asTensor(imgs_path, n_light, size)
light_p = torch.from_numpy(generate_light_p(7, 3.2, 1000))[light_idx]     # 世界坐标系下的

new_mask = camera.new_mask
albedo_ref = 0.3*torch.mean(imgs_tensor, dim=0)
depth_ref = torch.ones_like(albedo_ref.clone()[:,:,0])*plane_height
gamma = nn.Parameter(torch.tensor(1.5))
depth_0_delta = nn.Parameter(torch.zeros_like(depth_ref))
roughness_0 = texture_gd[:,:,6:9]
specular_0 = texture_gd[:,:,9:]


##############################################################################
##############################################################################
opt_gamma = torch.optim.Adam([gamma], lr = 200*lr)
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

result_0 = torch.zeros(n_light,size, size, 3)
plane_XYZ = camera.get_planeXYZ_fromAdepth(10)

for i in tqdm.tqdm(range(n_epoch)):
    if i<gamma_step:
        opt_gamma.zero_grad()
    elif i < depth_step:
        opt_adam.zero_grad()
    else:
        opt_SGD.zero_grad()
    # if i > 57:
    #     depth__ = 2*depth_0_delta+depth_base

    depth__ = depth_0_delta+depth_base
    imgXYZ = camera.get_imgXYZ_from_rays(depth__[rays_idx_[:,0], rays_idx_[:,1]][:, None], rays_idx_)
    normal_0 = camera.get_N_from_depth_dif(depth__)
    # img_l_d = light_p[:,None,:] - plane_XYZ.rehsape(-1,3)
    img_l_d = light_p[:,None,:] - imgXYZ


    textures = torch.cat([(normal_0+1)/2, torch.pow(albedo_ref, 1+gamma), roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
    # textures = torch.cat([(normal_0+1)/2, texture_gd[:,:,3:6], roughness_0, specular_0], dim=-1)[rays_idx_[:,0],rays_idx_[:,1],:]
    result = render_lsq(albedo_ref[rays_idx_[:,0],rays_idx_[:,1],:], (normal_0[rays_idx_[:,0],rays_idx_[:,1],:]+1)/2, img_l_d, gamma)
    # print(I_render.shape)
    # result = render_pixels(camera.vs_d[rays_idx_[:,0],rays_idx_[:,1],:], ggxrender, img_l_d, textures, light_int)#.reshape(n_light, size, size, 3)
    # print(result.shape)

    result_0[:,rays_idx_[:,0],rays_idx_[:,1],:] = result

    diff_img = torch.zeros_like(imgs_tensor)
    diff_img[:,camera.new_mask>0,:] = result_0[:,camera.new_mask>0] - torch.pow(imgs_tensor[:,camera.new_mask>0], 1/2.2)
    diff_img = torch.sum(torch.sum(torch.abs(diff_img), dim=0), dim=-1)[:,:,None].repeat(1,1,3)
    diff_img = diff_img / torch.max(diff_img)

    loss_img = L1Loss(result, torch.pow(imgs_tensor[:,rays_idx_[:,0], rays_idx_[:,1]], 1/2.2))
    loss_depth = Loss(depth__[camera.new_mask>0], depth_base[camera.new_mask>0])
    loss = loss_img + 1e2*loss_depth
    loss.backward()
    grad_img = torch.abs(depth_0_delta.grad)[:,:,None].repeat(1,1,3)
    grad_img = grad_img/torch.max(grad_img)
    print('\n', i, loss.item(), 1e2*loss_depth.item(), gamma.item())
    if i< gamma_step:
        opt_gamma.step()
    elif i< depth_step:
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
    row2 = np.hstack((result_0[0].detach().numpy(), diff_img.detach().numpy(), n_img,  torch.pow(albedo_ref, 1+gamma).detach().numpy()))
    # if i > 50:
    #     row1[rays_idx_[:,0], size+rays_idx_[:,1],:] = [0.,1.,0.]

    cv2.imwrite(out_path+'diff.png', (diff_img.detach().numpy()*255).astype(np.uint8))
    cv2.imwrite(out_path+'grad.png', (grad_img.detach().numpy()*255).astype(np.uint8))

    cv2.imwrite(out_path+'render_res.png', (np.vstack((row1, row2))*255).astype(np.uint8))
    cv2.imwrite(out_path+'n_img.png'.format(i), (n_img*255).astype(np.uint8))
    # np.save(out_path+'depth', (alpha*depth_0_delta).detach().numpy())
    gif_img.append((n_img*255).astype(np.uint8)[:,:,[2,1,0]])
    plt.plot(loss_list)
    plt.savefig(out_path+'loss.png')
    plt.close()

    # if i >50:
    #     rays_idx_ = select_rays(diff_img[:,:,0], n_rays)
    #     opt_adam.param_groups[0]['lr'] = 0.3*opt_adam.param_groups[0]['lr']


imageio.mimsave(out_path+'normal.gif', gif_img, fps=10)
np.save(out_path+'render_img.npy', torch.pow(result_0[0], 2.2).detach().numpy())
np.save(out_path+'depth.npy', (depth__).detach().numpy())
vis_depth_with_color((depth__).detach().numpy(), torch.pow(result_0[0], 2.2).detach().numpy(), camera.new_mask)