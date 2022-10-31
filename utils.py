from dataclasses import replace
from hashlib import new
from math import gamma

from torch import log_
import numpy as np
import cv2
import open3d as o3d
import torch
import os
import kornia
from shutil import copyfile
import helpers

def vis_depth(depth, mask):
    # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    x = np.arange(0,depth.shape[1])
    y = np.arange(0,depth.shape[0])
    X,Y = np.meshgrid(x/1000,y/1000)
    # print(pts.shape, X.shape, pts)
    pts = np.concatenate((X[None,:], Y[None,:], depth[None,:]), axis=0).transpose(1,2,0)
    pts = pts[mask>0.1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.5, 0.506, 0.5])
    o3d.visualization.draw([pcd])

def vis_depth_with_color(depth, render_img, mask):
    # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    x = np.arange(0,depth.shape[1])
    y = np.arange(0,depth.shape[0])
    X,Y = np.meshgrid(x/1000,y/1000)
    # print(pts.shape, X.shape, pts)
    pts = np.concatenate((X[None,:], Y[None,:], depth[None,:]), axis=0).transpose(1,2,0)
    pts = pts[mask>0.1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = render_img[mask>0.1].reshape(-1,3)[:,[2,1,0]]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.paint_uniform_color([0.5, 0.506, 0.5])
    o3d.visualization.draw([pcd])

def vis_pointscloud(pts):
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([1, 0.706, 0])          # 蓝 [0, 0.651, 0.929]
    o3d.visualization.draw([pcd,FOR1])


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

def render_img(vs_d, ggxrenderer, img_l_d, textures, lampIntensity=100):
    '''
        input:
            camera: Camera Classs
            renderer: GGXrenderer
            img_l_d: torch.size([bs, n_light, H, W, 3])
            textures: torch.size([2400, 3200, 12])
        output:
            render_image:
    '''
    n_light, H, W, ch = img_l_d.shape
    vs_d = vs_d.repeat(n_light,1,1,1)
    result = ggxrenderer(textures.reshape(-1,12), img_l_d.reshape(n_light,-1,3), -vs_d.reshape(n_light,-1,3), lampIntensity)
    result=torch.pow(result+ 1e-6, 1/2.2)
    return result.reshape(n_light, H, W, 3)

def render_pixels(vs_d, ggxrenderer, img_l_d, textures, lampIntensity=100):
    '''
        input:
            vs_d: torch.size([n_rays, 3])
            renderer: GGXrenderer
            img_l_d: torch.size([n_lights, n_rays, 3])
            textures: torch.size([n_rays, 12])
        output:
            render_pixels: [n_lights, n_rays, 3]
    '''
    n_light, n_rays, ch = img_l_d.shape
    vs_d = vs_d[None,:,:].repeat(n_light,1,1)
    result = ggxrenderer(textures, img_l_d, -vs_d, lampIntensity)
    # result=torch.pow(result+ 1e-6, 1/2.2)
    return result

def generate_light_p(h, r, n_light):
    '''
        input:
            n_light: 光源数量
    '''
    x = np.arange(n_light)
    y = np.arange(n_light)
    x = r*np.cos(np.pi/n_light*2*x)
    y = r*np.sin(np.pi/n_light*2*y)
    return np.concatenate([x[:, None], y[:, None], h*np.ones_like(x)[:, None]], axis=1).astype(np.float32)

def read_imgs_asTensor(imgs_path, n_imgs, size):
    '''
        input:
            imgs_path: 文件夹路径
            n_imgs: 从中选取的图片数量
        output:
    '''
    imgs_list = os.listdir(imgs_path)
    imgs_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    imgs_idx = np.random.choice(np.arange(len(imgs_list)), n_imgs, replace=False)
    new_list = [imgs_list[x] for x in imgs_idx]
    print(new_list)
    # imgs_list = np.random.choice(imgs_list, n_imgs, replace=False)
    # print(imgs_list)
    H = 2400; W= 3200
    return torch.pow(torch.cat([torch.from_numpy(cv2.imread(imgs_path + x)[None,:,:,:])/255 for x in new_list], dim=0), 2.2)[:, H//2-size//2:H//2-size//2+size, W//2-size//2:W//2-size//2+size], imgs_idx

def smooth_loss(depth, mask):
    loss3 = torch.sum(torch.abs(kornia.filters.laplacian(depth[None, None, :, :], 11)[0,0,mask>0]))
    loss11 = torch.sum(torch.abs(kornia.filters.laplacian(depth[None, None, :, :], 17)[0,0,mask>0]))
    loss31 = torch.sum(torch.abs(kornia.filters.laplacian(depth[None, None, :, :], 31)[0,0,mask>0]))
    print(loss3.item(), loss11.item()*0.1, loss31.item()*0.1)
    return loss3 + loss11*0.1 + loss31*0.1

def select_rays(diff, n_rays):
    h,w = diff.shape
    prob = torch.rand(h,w)*0.2+0.8
    select_p = prob*diff
    val, idx = select_p.reshape(-1).topk(k=n_rays)
    h_idx = idx//h; w_idx = idx%w
    return torch.cat((h_idx[:,None], w_idx[:,None]), dim=-1)

def code_backup(out_path, code_path):
    os.makedirs(os.path.join(out_path, "code_backup"), exist_ok=True)
    for f_name in os.listdir(code_path):
        if f_name[-3:] == ".py":
            copyfile(os.path.join(code_path, f_name), os.path.join(out_path+"code_backup", f_name))

def render_lsq(albdeo, N, L, light_int):
    '''
        input:
            albedo: [n_rays, 3]
            N: [n_rays, 3]
            L: [n_light, n_rays, 3]
            light_int
        output:
            I_render: [n_light, n_rays, 3]
    '''
    N_ = helpers.normalize(N)
    L_ = helpers.normalize(L)
    dis = L.norm(dim=-1)
    albdeo = albdeo[None,:,:]/dis[:,:,None]**2
    # print(albdeo.shape, N.shape, L.shape, torch.sum(N*L, dim=-1, keepdim=True).shape)
    return torch.clip(torch.pow(albdeo*torch.clip(torch.sum(N_*L_, dim=-1, keepdim=True), min=0.0)+1e-8, 1/2.2)*light_int, max=1.0)

def tone_mapping_img(img):
    '''
        input:
            img: [..., 3]
        output:
            tone_img: [..., 3]
        from paper: [2022] Extracting Triangular 3D Models, Materials, and Lighting From Images
    '''
    a = 0.055
    log_img = torch.log(img+1)
    tone_img = torch.zeros_like(log_img)
    tone_img[log_img<=0.0031308] = 12.92*log_img[log_img<=0.0031308]
    tone_img[log_img>0.0031308] = (1+a)*torch.pow(log_img[log_img>0.0031308], 1/2.4) - a
    return tone_img