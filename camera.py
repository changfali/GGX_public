import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np


class Camera(object):
    def __init__(self, W, H, K, W2C, mask):
        """
        W, H: int
        K, W2C: 4x4 tensor
        """
        self.W = W
        self.H = H
        self.K = K
        self.W2C = W2C
        self.K_inv = torch.inverse(K)
        self.C2W = torch.inverse(W2C)
        self.device = self.K.device

        self.uv = self.get_uv()
        self.mask0 = torch.from_numpy(mask)
        self.new_mask = self.get_NewMask()

        self.vs_o, self.vs_d, self.ray_d_norm = self.get_rays(self.uv)

    def get_uv(self):
        '''
        output:
            uv: u: uv[...,0], v: uv[...,1]
        '''
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        uv = torch.from_numpy(np.stack((u, v), axis=-1).astype(np.float32)) + 0.5
        return uv

    def get_NewMask(self):
        '''
            output:
                mask_new: [H, W] 剔除输入mask的一圈轮廓点
        '''
        si = 21
        k = torch.ones(1,1,si,si, dtype=torch.float64)
        mask0 = self.mask0[None,:,:].unsqueeze(0)
        mask_new = F.conv2d(mask0, nn.Parameter(k, requires_grad=False), padding=(si-1)//2)
        mask_new = mask_new[0,0]
        mask_new[mask_new<si*si*0.9]=0.; mask_new[mask_new>0]=1

        return mask_new

    def crop_region(self, trgt_W, trgt_H, center_crop=False, ul_corner=None, image=None):
        K = self.K.clone()
        if ul_corner is not None:
            ul_col, ul_row = ul_corner
        elif center_crop:
            ul_col = self.W // 2 - trgt_W // 2
            ul_row = self.H // 2 - trgt_H // 2
        else:
            ul_col = np.random.randint(0, self.W - trgt_W)
            ul_row = np.random.randint(0, self.H - trgt_H)
        # modify K
        K[0, 2] -= ul_col
        K[1, 2] -= ul_row

        camera = Camera(trgt_W, trgt_H, K, self.W2C.clone())

        if image is not None:
            assert image.shape[0] == self.H and image.shape[1] == self.W, "image size does not match specfied size"
            image = image[ul_row : ul_row + trgt_H, ul_col : ul_col + trgt_W]
        return camera, image

    def resize(self, factor, image=None):
        trgt_H, trgt_W = int(self.H * factor), int(self.W * factor)
        K = self.K.clone()
        K[0, :3] *= trgt_W / self.W
        K[1, :3] *= trgt_H / self.H
        camera = Camera(trgt_W, trgt_H, K, self.W2C.clone())

        if image is not None:
            device = image.device
            image = cv2.resize(image.detach().cpu().numpy(), (trgt_W, trgt_H), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(image).to(device)
        return camera, image

    def get_rays(self, uv):
        """
        uv: [..., 2]
        """
        dots_sh = list(uv.shape[:-1])

        uv = uv.view(-1, 2)
        uv = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1)
        ray_d = torch.matmul(
            torch.matmul(uv, self.K_inv[:3, :3].transpose(1, 0)),
            self.C2W[:3, :3].transpose(1, 0),
        ).reshape(
            dots_sh
            + [
                3,
            ]
        )

        ray_d_norm = ray_d.norm(dim=-1)
        ray_d = ray_d / ray_d_norm.unsqueeze(-1)

        ray_o = (
            self.C2W[:3, 3]
            .unsqueeze(0)
            .expand(uv.shape[0], -1)
            .reshape(
                dots_sh
                + [
                    3,
                ]
            )
        )
        return ray_o, ray_d, ray_d_norm

    def get_camera_origin(self, prefix_shape=None):
        ray_o = self.C2W[:3, 3]
        if prefix_shape is not None:
            prefix_shape = list(prefix_shape)
            ray_o = ray_o.view([1,] * len(prefix_shape) + [3,]).expand(
                prefix_shape
                + [
                    3,
                ]
            )
        return ray_o

    def get_depthMap(self, pointscloud):
        '''
        点云转深度, 深度图会有空洞出现
            input:
                pointscloud: [n, 3]  # 相机坐标系下!!!!!
            output:
                img_z: [H,W] 带有空洞
        '''
        eps = 1e-10
        valid = pointscloud[:, 2]>eps
        z = pointscloud[valid, 2]
        uv = self.project(pointscloud[valid,:], round = True)

        valid = torch.bitwise_and(torch.bitwise_and((uv[:,0] >= 0), (uv[:,0] < self.W)),
                        torch.bitwise_and((uv[:,1] >= 0), (uv[:,1] < self.H)))
        uv = uv[valid,:]; z = z[valid]

        img_z = torch.tensor(np.full((self.H, self.W), np.inf).astype(np.float32))
        for uvi, zi in zip(uv, z):
            img_z[uvi[1], uvi[0]] = torch.min(img_z[uvi[1], uvi[0]], zi)

        img_z_shift = torch.cat([torch.unsqueeze(img_z, axis=-1), \
                        torch.unsqueeze(torch.roll(img_z, 1, 0), axis=-1), \
                        torch.unsqueeze(torch.roll(img_z, -1, 0), axis=-1), \
                        torch.unsqueeze(torch.roll(img_z, 1, 1), axis=-1), \
                        torch.unsqueeze(torch.roll(img_z, -1, 1), axis=-1)], dim=-1)

        img_z, idx = torch.min(img_z_shift, dim=-1)
        return img_z

    def get_full_depth_cv2(self, depth):
        '''
            input:
                depth: 带有空洞的深度图, 空洞处为0
            output:
                closing: 填充后的深度图, mask之外的深度为0, 即默认该点在相机光心位置
        '''
        depth = depth.clone().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel, iterations=3)
        closing[self.mask0.numpy()<1] = 0
        return torch.from_numpy(closing)

    # def get_N_from_depth(self, depth):
    #     '''
    #         input:
    #             depth: [H,W]
    #             mask: [H,W] 有效区域的mask
    #         output:
    #             N_: [H,W,3] 法线, 新的有效区域内
    #             new_mask: [H, W] 新的有效区域mask
    #     '''
    #     depth[self.mask0<1]=0
    #     depth[self.mask0>1] = torch.log(depth[self.mask0>1])

    #     u_f = torch.roll(depth,-1,1)
    #     v_f = torch.roll(depth,-1,0)
    #     u_f[self.new_mask<1]=0; v_f[self.new_mask<1]=0
    #     d_0 = torch.clone(depth); d_0[self.new_mask<1]=0

    #     du = u_f-d_0
    #     dv = v_f-d_0

    #     Nx = du*self.K[0,0]
    #     Ny = dv*self.K[1,1]
    #     Nz = -du*self.uv[:,:,0]-dv*self.uv[:,:,1]-1
    #     dz = torch.sqrt(Nx**2+Ny**2+Nz**2)

    #     N_ = torch.cat([(Nx/dz)[:,:,None], (Ny/dz)[:,:,None], (Nz/dz)[:,:,None]], dim=-1)
    #     N_[:,:,1:] = -N_[:,:,1:]
    #     N_[self.new_mask<1]=0
    #     return N_

    def get_N_from_depth_dif(self, depth):
        '''
            input:
                depth: [H,W]
            output:
                N_: [H,W,3] 法线, 新的有效区域内
                new_mask: [H, W] 新的有效区域mask
        '''
        depth = torch.log(depth)
        u_f = torch.roll(depth,-1,1)
        v_f = torch.roll(depth,-1,0)

        du = u_f-depth
        dv = v_f-depth

        Nx = du*self.K[0,0]
        Ny = dv*self.K[1,1]
        Nz = -du*self.uv[:,:,0]-dv*self.uv[:,:,1]-1
        dz = torch.sqrt(Nx**2+Ny**2+Nz**2)
        N_ = torch.cat([(Nx/dz)[:,:,None], (Ny/dz)[:,:,None], (Nz/dz)[:,:,None]], dim=-1)
        N_[:,:,1:] = -N_[:,:,1:]
        N_[self.new_mask<1]=0
        return N_

    def write_N_img(self, N_, out_path):
        n_ = ((N_.numpy()+1)/2*255).astype(np.uint8)[:,:,[2,1,0]]
        n_[self.new_mask.numpy()<1]=0
        cv2.imwrite(out_path+'normal100.png',n_)

    def get_imgXYZ_from_depth(self, depth0, pc=False):
        '''
            input:
                depth: 填充后的深度图pc=False, 或者pc=True点云(必须在相机坐标系下)
            output:
                imgxyz: [H,W,3] 每个像素点在世界坐标系下的XYZ, mask之外的点在填充深度时置为0(即
                        默认在相机光心位置), 使用时需注意使用self.new_mask去除
        '''
        if pc:
            depth1 = self.get_depthMap(depth0)
            depth1 = torch.where(torch.isinf(depth1), torch.full_like(depth1, 0), depth1)
            depth = self.get_full_depth_cv2(depth1)
        else:
            depth = depth0.clone()
        uv = self.uv
        dots_sh = list(uv.shape[:-1])
        uv = uv.view(-1, 2)
        uv = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1)

        uv_z = torch.cat((uv*depth.reshape(-1,1), torch.ones_like(uv[..., 0:1])), dim=-1)
        xyz = torch.matmul(torch.matmul(uv_z, self.K_inv.transpose(1, 0)), self.C2W.transpose(1,0))[:,:3]

        if pc:
            return xyz.reshape(dots_sh + [3]), depth
        else:
            return xyz.reshape(dots_sh + [3])

    def get_imgXYZ_from_rays(self, depth, rays_idx):
        '''
            input:
                depth: [n_rays, 1]填充后的深度图
                rays_idx: [n_rays, 2]光线对应的图片像素索引
            output:
                imgxyz: [n_rays, 3] 每条光线对应的像素点在世界坐标系下的XYZ
        '''
        uv = self.uv[rays_idx[:, 0], rays_idx[:, 1]]
        uv = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1)

        uv_z = torch.cat((uv*depth, torch.ones_like(uv[..., 0:1])), dim=-1)
        xyz = torch.matmul(torch.matmul(uv_z, self.K_inv.transpose(1, 0)), self.C2W.transpose(1,0))[:,:3]

        return xyz


    def get_planeXYZ_fromAdepth(self, Adepth):
        '''
            input:
                Adepth: 相机坐标系下的深度
            output:
                imgXYZ_Plane: [H, W, 3], 世界坐标系下
        '''
        uv = self.uv
        dots_sh = list(uv.shape[:-1])
        uv = uv.view(-1, 2)
        uv = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1)
        depth_pl = torch.ones(self.H, self.W)*Adepth

        uv_z = torch.cat((uv*depth_pl.reshape(-1,1), torch.ones_like(uv[..., 0:1])), dim=-1)
        xyz = torch.matmul(torch.matmul(uv_z, self.K_inv.transpose(1, 0)), self.C2W.transpose(1,0))[:,:3]
        return xyz.reshape(dots_sh + [3])

    def project(self, points, round = False):
        """
        input:
            points: [..., 3]  # 相机坐标系下
            round: 是否取整
        output:
            uv, u: uv[...,0], v: uv[...,1]
        """
        dots_sh = list(points.shape[:-1])

        points = points.view(-1, 3)
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        # uv = torch.matmul(
        #     torch.matmul(points, self.W2C.transpose(1, 0)),
        #     self.K.transpose(1, 0),
        # )
        uv = torch.matmul(
            points,
            self.K.transpose(1, 0),
        )
        uv = uv[:, :2] / uv[:, 2:3]

        uv = uv.view(
            dots_sh
            + [
                2,
            ]
        )
        if round:
            return uv.floor().long()
        else:
            return uv

    def World_2_Camera(self, pts):
        '''
            input:
                pts: [..., 3]           世界坐标系下的点
            output:
                pts_C: [..., 3]         相机坐标系下的点
        '''
        dots_sh = list(pts.shape[:-1])

        pts = pts.view(-1, 3)
        pts_ = torch.cat((pts, torch.ones_like(pts[:, 0:1])), dim=-1)
        pts_C = torch.matmul(pts_, self.W2C.transpose(1, 0))[:,:3].reshape(dots_sh+[3])
        return pts_C

    def Camera_2_World(self, pts):
        '''
            input:
                pts: [..., 3]           相机坐标系下的点
            output:
                pts_W: [..., 3]         世界坐标系下的点
        '''
        dots_sh = list(pts.shape[:-1])

        pts = pts.view(-1, 3)
        pts_ = torch.cat((pts, torch.ones_like(pts[:, 0:1])), dim=-1)
        pts_W = torch.matmul(pts_, self.C2W.transpose(1, 0))[:,:3].reshape(dots_sh+[3])
        return pts_W


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import open3d as o3d
    from utils import *

    k = torch.tensor([[3602, 0.,   1600, 0.],
                    [0.,   3602, 1200, 0.],
                    [0.,   0.,   1., 0.],
                    [0.,   0.,   0., 1.]])
    W2C = torch.tensor([[1., 0.,  0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0.,  -1., 10],
                        [0., 0.,  0.,  1]])
    mask = cv2.imread(r'test_data\mask.png')[:,:,0]/255
    camera = Camera(3200,2400,k,W2C,mask)

    pc = torch.from_numpy(np.loadtxt(r'test_data\fliter_MVS_pc.txt').astype(np.float32))[:,:3]
    print(torch.max(pc[:,2]), torch.min(pc[:,2]))

    ######
    imgxyz, depth = camera.get_imgXYZ_from_depth(pc, pc=True)       # 将相机坐标系下的点云或者深度图转世界坐标系下的XYZ坐标
    # vis_pointscloud(imgxyz[camera.new_mask>0,:].reshape(-1,3))
    print(imgxyz.shape, depth.shape)
    np.save('out/depth_full000', depth.numpy())
    np.save('out/imgxyz', imgxyz.numpy())

    ######
    Normal_map = camera.get_N_from_depth_dif(depth)
    n_ = ((Normal_map.numpy()+1)/2*255).astype(np.uint8)[:,:,[2,1,0]]
    n_[camera.new_mask.numpy()<1]=0
    cv2.imwrite('out/normal1.png',n_)
    cv2.imwrite('out/mask0.png',(camera.mask0.numpy()*255).astype(np.uint8))
    cv2.imwrite('out/new_mask.png',(camera.new_mask.numpy()*255).astype(np.uint8))

    ########
    mask0 = cv2.imread(r'out\mask0.png')[:,:,0]/255
    new_mask = cv2.imread(r'out\new_mask.png')[:,:,0]/255
    plt.imshow(mask0 - new_mask)
    plt.show()

    vis_depth(np.load(r'out\depth_full000.npy'), mask0)
