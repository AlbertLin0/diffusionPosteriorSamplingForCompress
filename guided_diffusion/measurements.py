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


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 



from models.elic import TestModel as ELICModel

elic_paths = [
    'models/ELIC_0008_ft_3980_Plateau.pth.tar',
    'models/ELIC_0016_ft_3980_Plateau.pth.tar',
    'models/ELIC_0032_ft_3980_Plateau.pth.tar',
    'models/ELIC_0150_ft_3980_Plateau.pth.tar',
    'models/ELIC_0450_ft_3980_Plateau.pth.tar',
]

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
    'models/bmshj2018-hyperprior-1-7eb97409.pth.tar',
    'models/bmshj2018-hyperprior-2-93677231.pth.tar',
    'models/bmshj2018-hyperprior-3-6d87be32.pth.tar',
    'models/bmshj2018-hyperprior-4-de1b779c.pth.tar',
    'models/bmshj2018-hyperprior-5-f8b614e1.pth.tar',
    'models/bmshj2018-hyperprior-6-1ab9c41e.pth.tar',
    'models/bmshj2018-hyperprior-7-3804dcbd.pth.tar',
    'models/bmshj2018-hyperprior-8-a583f0cf.pth.tar',
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
@register_operator(name='bmjshj2018_factorized')
class CodecOperator(NonLinearOperator):
    def __init__(self, q, device):
        self.codec = bmshj2018_factorized(quality=q, metric='mse', pretrained=True, progress=True)
        self.codec = self.codec.cuda()
        self.codec.eval()
        print("load gg18 q: {}".format(q))
        
    def forward(self, data, **kwargs):
        out = self.codec.forward((data + 1.0) / 2.0)
        return (out["x_hat"]*2.0) - 1.0
    
    def encode(self, data, **kwargs):
        y_hat = self.codec.encode((data+ 1.0) / 2.0)
        return y_hat
    
    def decode(self, y_hat, **kwargs):
        return self.codec.decode(y_hat)        

    def getBpp(self, data, **kwargs):
        out = self.codec.compress((data + 1.0) / 2.0)
        byte_len = len(out['strings'][0][0])
        return byte_len * 8 /(256*256*3)

    def y_hat_bpp(self, data):
        return self.codec.y_hat_bpp(data)
    
    def compress(self, data):
        return self.codec.compress((data+ 1.0) / 2.0)
    
    
@register_operator(name='bmshj2018_hyperprior')
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
    



