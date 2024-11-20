'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)




@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

from models.elic import TestModel as ELICModel

elic_paths = [
    'bins/ELIC_0008_ft_3980_Plateau.pth.tar',
    'models/ELIC_0016_ft_3980_Plateau.pth.tar',
    'models/ELIC_0032_ft_3980_Plateau.pth.tar',
    'models/ELIC_0150_ft_3980_Plateau.pth.tar',
    'bins/ELIC_0450_ft_3980_Plateau.pth.tar',
]

# def addNoise(y_n):
#     return y_n + torch.randn_like(y_n, device=y_n.device) * 0.05

@register_operator(name='elic')
class CodecOperator(NonLinearOperator):
    def __init__(self, device, q=2):
        self.elic = ELICModel()
        self.elic.load_state_dict(torch.load(elic_paths[q-1]))
        self.elic = self.elic.cuda()
        self.elic.eval()
        print("load elic: {}".format(elic_paths[q-1]))
        
    def forward(self, data, **kwargs):
        enc_out = self.elic((data + 1.0) / 2.0, "enc", False)
        dec_out = self.elic(enc_out["y_hat"], "dec", False)
        return (dec_out["x_bar"] * 2.0) - 1.0

    def getBpp(self, data, **kwargs):
        enc_out = self.elic((data + 1.0) / 2.0, "enc", False)
        y_bpp = torch.mean(torch.sum(-torch.log2(enc_out["likelihoods"]["y"]),dim=(1,2,3)), dim=0)
        z_bpp = torch.mean(torch.sum(-torch.log2(enc_out["likelihoods"]["z"]),dim=(1,2,3)), dim=0)
        return (y_bpp.item() + z_bpp.item()) / 255.0 / 255.0 /3

gg18_paths = [
    'bins/bmshj2018-hyperprior-1-7eb97409.pth.tar',
    'models/bmshj2018-hyperprior-2-93677231.pth.tar',
    'bins/bmshj2018-hyperprior-3-6d87be32.pth.tar',
    'bins/bmshj2018-hyperprior-4-de1b779c.pth.tar',
    'bins/bmshj2018-hyperprior-5-f8b614e1.pth.tar',
    'bins/bmshj2018-hyperprior-6-1ab9c41e.pth.tar',
    'bins/bmshj2018-hyperprior-7-3804dcbd.pth.tar',
    'bins/bmshj2018-hyperprior-8-a583f0cf.pth.tar',
]

Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]
from models.gg18 import ScaleHyperpriorSTE
@register_operator(name='gg18')
class CodecOperator(NonLinearOperator):
    def __init__(self, q, device):
        self.codec = ScaleHyperpriorSTE(Ns[q-1], Ms[q-1])
        self.codec.load_state_dict_gg18(torch.load(gg18_paths[q - 1]))
        self.codec = self.codec.cuda()
        self.codec.eval()
        print("load gg18 q: {}".format(q))
        
    def forward(self, data, **kwargs):
        out = self.codec.forward((data + 1.0) / 2.0)
        return (out["x_hat"] * 2.0) - 1.0     
     
    def getBpp(self, data, **kwargs):
        out = self.codec.compress((data + 1.0) / 2.0)
        byte_len = len(out['strings'][0][0])
        return byte_len * 8 /(256*256*3)

import numpy as np
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior
@register_operator(name='gg18_test')
class CodecOperator(NonLinearOperator):
    def __init__(self, q, device):
        self.codec = bmshj2018_factorized(quality=q, metric='mse', pretrained=True, progress=True)
        self.codec = self.codec.cuda()
        self.codec.eval()
        print("load gg18 q: {}".format(q))
        
    def forward(self, data, **kwargs):
        out = self.codec.forward((data + 1.0) / 2.0)
        return (out["x_hat"]*2.0) - 1.0
    
    def getBpp(self, data, **kwargs):
        out = self.codec.compress((data + 1.0) / 2.0)
        byte_len = len(out['strings'][0][0])
        return byte_len * 8 /(256*256*3)
    
@register_operator(name='gg18_zoo')
class CodecOperator(NonLinearOperator):
    def __init__(self, q, device):
        self.codec = bmshj2018_hyperprior(quality=q, metric='mse', pretrained=True, progress=True)
        self.codec = self.codec.cuda()
        self.codec.eval()
        print("load gg18 q: {}".format(q))
        
    def forward(self, data, **kwargs):
        out = self.codec.forward((data + 1.0) / 2.0)
        return (out["x_hat"]*2.0) - 1.0
    
    def getBpp(self, data, **kwargs):
        out = self.codec.compress((data + 1.0) / 2.0)
        byte_len = len(out['strings'][0][0])
        return byte_len * 8 /(256*256*3)
    

# =============
# DCVC
# =============
from models.DCVC_net import DCVC_net
import json
def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()

def filter_dict(result):
    keys = ['i_frame_num', 'p_frame_num', 'ave_i_frame_bpp', 'ave_i_frame_quality', 'ave_p_frame_bpp',
            'ave_p_frame_bpp_mv_y', 'ave_p_frame_bpp_mv_z', 'ave_p_frame_bpp_y',
            'ave_p_frame_bpp_z', 'ave_p_frame_quality', 'ave_all_frame_bpp', 'ave_all_frame_quality']
    res = {k: v for k, v in result.items() if k in keys}
    return res

@register_operator(name='DCVC')
class OperateDCVC():
    def __init__(self, device, q=2):
        self.video_net = DCVC_net()
        load_checkpoint = torch.load('models/model_dcvc_quality_0_psnr.pth', map_location=device)
        self.video_net.load_state_dict(load_checkpoint)
        self.video_net.to(device)
        self.video_net.eval()

        self.elic = ELICModel()
        self.elic.load_state_dict(torch.load(elic_paths[q-1], map_location=device))
        self.elic.to(device)
        self.elic.eval()
        print('load model successfully!')

        self.gop_size = 10
        self.frame_num = 105

        self.frame_types = []
        self.qualitys = []
        self.bits = []
        self.bits_mv_y = []
        self.bits_mv_z = []
        self.bits_y = []
        self.bits_z = []


    def forward(self, data, flag = 0,**Kwargs):
        ref_frame = None
        # Kwargs['ori_frame'] = data
        frame_pixel_num = data.shape[2]*data.shape[3]
        if Kwargs['frame_idx'] % self.gop_size == 0:
            self.frame_types.append(0)
            enc_out = self.elic((data+1.0)/2.0, "enc", False)
            dec_out = self.elic(enc_out["y_hat"], "dec", False)
            return (dec_out["x_bar"] *2.0) - 1.0
        else:
            result = self.video_net((Kwargs['ref_frame'] + 1.0) / 2, (data + 1.0) / 2.0, flag = flag)
            ref_frame = result['recon_image']
            bpp = result['bpp']
            self.frame_types.append(1)
            self.bits.append(bpp.item() * frame_pixel_num)
            # self.bits_mv_y.append(result['bpp_mv_y'].item() * frame_pixel_num)
            # self.bits_mv_z.append(result['bpp_mv_z'].item() * frame_pixel_num)
            # self.bits_y.append(result['bpp_y'].item() * frame_pixel_num)
            # self.bits_z.append(result['bpp_z'].item() * frame_pixel_num)

            # print("frame index: ", Kwargs['frame_idx'])
            # print("bpp: ", bpp)
            # print("bits_mv_y: ", result['bpp_mv_y'].item() * frame_pixel_num)
            # print("bits_mv_z: ", result['bpp_mv_z'].item() * frame_pixel_num)
            # print("bits_y: ", result['bpp_y'].item() * frame_pixel_num)
            # print("bits_z: ", result['bpp_z'].item() * frame_pixel_num)

            # ref_frame = ref_frame.clamp_(0, 1)
            # self.qualitys.append(PSNR(ref_frame, (data + 1.0) / 2.0))

            # print("frame index: ", Kwargs['frame_idx'])
            # print("bpp: ", bpp)
            # print("psnr: ", PSNR(ref_frame, (data + 1.0) / 2.0))
            return (ref_frame * 2.0) - 1.0

    def getBpp(self, data, **Kwargs):
        """ """
        frame_pixel_num = data.shape[2]*data.shape[3]
        if Kwargs['frame_idx'] % self.gop_size == 0:
            enc_out = self.elic((data+1.0)/2.0, "enc", False)
            y_bpp = torch.mean(torch.sum(-torch.log2(enc_out["likelihoods"]["y"]),dim=(1,2,3)), dim=0) / frame_pixel_num
            z_bpp = torch.mean(torch.sum(-torch.log2(enc_out["likelihoods"]["z"]),dim=(1,2,3)), dim=0) / frame_pixel_num
            return y_bpp.item() + z_bpp.item()
        else: 
            result = self.video_net((Kwargs['ref_frame'] + 1.0) / 2, (data + 1.0) / 2.0, flag = 1)
            bpp = result['bpp']
            return bpp.item()


    def result(self):
        cur_all_i_frame_bit = 0
        cur_all_i_frame_quality = 0
        cur_all_p_frame_bit = 0
        cur_all_p_frame_bit_mv_y = 0
        cur_all_p_frame_bit_mv_z = 0
        cur_all_p_frame_bit_y = 0
        cur_all_p_frame_bit_z = 0
        cur_all_p_frame_quality = 0
        cur_i_frame_num = 0
        cur_p_frame_num = 0
        for idx in range(self.frame_num):
            if self.frame_types[idx] == 0:
                cur_all_i_frame_bit += self.bits[idx]
                cur_all_i_frame_quality += self.qualitys[idx]
                cur_i_frame_num += 1
            else:
                cur_all_p_frame_bit += self.bits[idx]
                cur_all_p_frame_bit_mv_y += self.bits_mv_y[idx]
                cur_all_p_frame_bit_mv_z += self.bits_mv_z[idx]
                cur_all_p_frame_bit_y += self.bits_y[idx]
                cur_all_p_frame_bit_z += self.bits_z[idx]
                cur_all_p_frame_quality += self.qualitys[idx]
                cur_p_frame_num += 1
        log_result = {}
        frame_pixel_num = 256*256
        log_result['frame_pixel_num'] = frame_pixel_num
        log_result['i_frame_num'] = cur_i_frame_num
        log_result['p_frame_num'] = cur_p_frame_num
        log_result['ave_i_frame_bpp'] = cur_all_i_frame_bit / cur_i_frame_num / frame_pixel_num
        log_result['ave_i_frame_quality'] = cur_all_i_frame_quality / cur_i_frame_num
        if cur_p_frame_num > 0:
            total_p_pixel_num = cur_p_frame_num * frame_pixel_num
            log_result['ave_p_frame_bpp'] = cur_all_p_frame_bit / total_p_pixel_num
            log_result['ave_p_frame_bpp_mv_y'] = cur_all_p_frame_bit_mv_y / total_p_pixel_num
            log_result['ave_p_frame_bpp_mv_z'] = cur_all_p_frame_bit_mv_z / total_p_pixel_num
            log_result['ave_p_frame_bpp_y'] = cur_all_p_frame_bit_y / total_p_pixel_num
            log_result['ave_p_frame_bpp_z'] = cur_all_p_frame_bit_z / total_p_pixel_num
            log_result['ave_p_frame_quality'] = cur_all_p_frame_quality / cur_p_frame_num
        else:
            log_result['ave_p_frame_bpp'] = 0
            log_result['ave_p_frame_quality'] = 0
            log_result['ave_p_frame_bpp_mv_y'] = 0
            log_result['ave_p_frame_bpp_mv_z'] = 0
            log_result['ave_p_frame_bpp_y'] = 0
            log_result['ave_p_frame_bpp_z'] = 0
        log_result['ave_all_frame_bpp'] = (cur_all_i_frame_bit + cur_all_p_frame_bit) / \
                                          (self.frame_num * frame_pixel_num)
        log_result['ave_all_frame_quality'] = (cur_all_i_frame_quality + cur_all_p_frame_quality) / self.frame_num

        res = filter_dict(log_result)
        with open('./result_qianyi.json', 'w') as fp:
            json.dump(res, fp, indent=2)


# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
    



