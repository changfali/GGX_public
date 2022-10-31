import torch
import torch.nn as nn
import numpy as np
import os
import helpers

PI = 3.141592653589793

class GGXRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda

    def compute_diffuse(self, diffuse, specular):
        return diffuse * (1.0 - specular) / PI

    #Compute the distribution function D driving the statistical orientation of the micro facets.
    def compute_distribution(self, roughness, NdotH):
        alpha = torch.square(roughness)
        underD = 1/torch.clamp((torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0), min = 0.001)
        return (torch.square(alpha * underD)/PI)

    #Compute the fresnel approximation F
    def compute_fresnel(self, specular, VdotH):
        sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
        return specular + (1.0 - specular) * sphg

    # def compute_fresnel(self, specular, VdotH):
    #     return specular + (1.0 - specular) * torch.pow(1-VdotH, 5)


    #Compute the Geometry term (also called shadowing and masking term) G taking into account how microfacets can shadow each other.
    # def compute_geometry(self, roughness, NdotL, NdotV):
    #     return self.G1(NdotL, torch.square(roughness)/2) * self.G1(NdotV, torch.square(roughness)/2)

    # def G1(self, NdotW, k):
    #     return 1.0/torch.clamp((NdotW * (1.0 - k) + k), min = 0.001)

    def compute_geometry(self, roughness, NdotL, NdotV):
        k = torch.square(roughness+1)/8
        GL = NdotL/(NdotL*(1-k)+k)
        GV = NdotV/(NdotV*(1-k)+k)
        return GL*GV

    def calculateBRDF(self, svbrdf, wiNorm, woNorm):
        '''
            svbrdf: [h, w, 12], normals(-1,1), diffuse(0,1), roughness(0,1), specular(0,1) with 3 channels each TODO: 粗糙度的通道数是否要简化
            wiNorm/woNorm: [nbrender, h, w, 3]
        '''
        h = helpers.normalize(torch.add(wiNorm, woNorm) / 2.0)

        normals = svbrdf[:,0:3].unsqueeze(0)
        normals = normals*2-1
        diffuse = svbrdf[:,3:6].unsqueeze(0)
        roughness = torch.clamp(svbrdf[:,6:9], min=0.001).unsqueeze(0)     # avoid division by 0
        specular = svbrdf[:,9:12].unsqueeze(0)

        NdotH = helpers.DotProduct(normals, h)
        NdotL = helpers.DotProduct(normals, wiNorm)
        NdotV = helpers.DotProduct(normals, woNorm)
        VdotH = helpers.DotProduct(woNorm, h)

        diffuse_rendered = self.compute_diffuse(diffuse, specular)
        D_rendered = self.compute_distribution(roughness, torch.clamp(NdotH, min = 0.0))
        G_rendered = self.compute_geometry(roughness, torch.clamp(NdotL,min = 0.0), torch.clamp(NdotV, min = 0.0))
        F_rendered = self.compute_fresnel(specular, torch.clamp(VdotH, min = 0.0))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered

        result = result + diffuse_rendered
        return result, NdotL

    def forward(self, svbrdf, wi, wo, lampIntensity, lossRendering = True, isAmbient = False, useAugmentation = True):
        '''
            input:
                svbrdf: [H, W, ch]
                wi: [n_lights, H, W, 3]
                w0: [n_lights, H, W, 3]

        '''
        wiNorm = helpers.normalize(wi)
        woNorm = helpers.normalize(wo)

        #Calculate how the image should look like with completely neutral lighting
        result, NdotL = self.calculateBRDF(svbrdf, wiNorm, woNorm)
        resultShape = result.shape
        lampIntensity = lampIntensity

        if not lossRendering:
            #If we are not rendering for the loss
            if not isAmbient:
                if useAugmentation:
                    #The augmentations will allow different light power and exposures
                    stdDevWholeBatch = torch.exp(torch.randn((), mean = -2.0, stddev = 0.5))
                    #add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                    lampIntensity = torch.abs(torch.randn((resultShape[0], resultShape[1], 1, 1, 1), mean = 10.0, stddev = stdDevWholeBatch)) # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                    #autoExposure
                    autoExposure = torch.exp(torch.randn((), mean = np.log(1.5), stddev = 0.4))
                    lampIntensity = lampIntensity * autoExposure
                else:
                    lampIntensity = torch.reshape(torch.FloatTensor(13.0), [1, 1, 1, 1, 1]) #Look at the renderings when not using augmentations
            else:
                #If this uses ambient lighting we use much small light values to not burn everything.
                if useAugmentation:
                    lampIntensity = torch.exp(torch.randn((resultShape[0], 1, 1, 1, 1), mean = torch.log(0.15), stddev = 0.5)) #No need to make it change for each rendering.
                else:
                    lampIntensity = torch.reshape(torch.FloatTensor(0.15), [1, 1, 1, 1, 1])
            #Handle light white balance if we want to vary it..
            if useAugmentation and not isAmbient:
                whiteBalance = torch.abs(torch.randn([resultShape[0], resultShape[1], 1, 1, 3], mean = 1.0, stddev = 0.03))
                lampIntensity = lampIntensity * whiteBalance

        lampFactor = lampIntensity * PI

        # if not isAmbient:
        #     if not lossRendering:
        #         #Take into accound the light distance (and the quadratic reduction of power)
        # lampDistance = torch.sqrt(torch.sum(torch.square(wi), axis = -1, keep_dims=True))
        lampDistance = wi.norm(dim=-1)
        lampFactor = lampFactor * helpers.lampAttenuation_pbr(lampDistance.unsqueeze(-1))

        result = result * lampFactor
        result = result * torch.clamp(NdotL, min = 0.0)

        if lossRendering:
            result = result / torch.unsqueeze(torch.clamp(wiNorm[:,:,2], min = 0.001), axis=-1)# This division is to compensate for the cosinus distribution of the intensity in the rendering.

        return torch.clamp(result, min=0, max=1)

if __name__ == "__main__":
    import numpy as np
    import cv2
    import torch
    import torch.nn as nn
    import skimage.io as io

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

    n_light = 32
    size = 3200

    textures = get_texture(r'test_data\textures\texture_only\metal_rusty_tj3rdiobw\tex/')[:2400,:3200]
    print('textures:', textures.shape)
    mask = cv2.imread(r'test_data\mask.png')[:,:,0]/255
    # pc = torch.from_numpy(np.loadtxt(r'test_data\fliter_MVS_pc.txt').astype(np.float32))[:,:3]

    camera = Camera(3200,2400,k,W2C,mask)
    ggxrender = GGXRenderer(use_cuda=False)

    ###################################################################
    plane_XYZ = camera.get_planeXYZ_fromAdepth(10)   # 世界坐标系下

    # light_p = torch.tensor([[0,0,10]]).repeat(n_light,1)# 世界坐标系下
    light_p = torch.from_numpy(generate_light_p(7, 3.2, 1000))

    N_light_batch = 2
    light_p_split = torch.split(light_p, N_light_batch, 0)

    textures[:,:, :2] = 0.5; textures[:,:,2]=1
    print(textures.shape)

    for j, lp in enumerate(light_p_split):
        img_l_d = lp[:,None, None,:] - plane_XYZ
        result = render_img(camera.vs_d, ggxrender, img_l_d, textures, 300)
        print(result.shape)
        for i in range(result.shape[0]):
            print(i)
            io.imsave(r'out\debug\1008/img_{}.png'.format(int(j*N_light_batch+i)), ((result[i].reshape(camera.H,camera.W,3).numpy())*255).astype(np.uint8)[:,:,[2,1,0]])