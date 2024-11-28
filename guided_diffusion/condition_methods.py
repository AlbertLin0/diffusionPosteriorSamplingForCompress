from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
# from util.img_utils import clear_color
import os
import numpy as np
__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.i = 999
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            
            if kwargs['flag'] == 'DPS':
                if self.i < 0:
                    difference = kwargs['true_measurement'] - x_0_hat
                else:
                    difference = measurement - self.operator.forward(data=x_0_hat, **kwargs)
                    # difference = self.operator.decode(measurement) - self.operator.forward(data=x_0_hat, **kwargs)
                norm = torch.linalg.norm(difference)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
                return norm_grad, norm
            else: 

                if self.i < 0:
                    difference = kwargs['true_measurement'] - x_0_hat
                else:
                    difference = self.operator.decode(measurement) - self.operator.forward(data=x_0_hat, **kwargs)
                    # difference = self.operator.forward(data=x_0_hat, **kwargs)

                norm = torch.linalg.norm(difference)
                # x_prev.grad.zeros()
                norm.backward()
                x_prev_grad = x_prev.grad.detach_()
                measurement.grad.zero_()
                # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, retain_graph=True, allow_unused=True)[0]
                tmp = x_prev - kwargs['xt_truth']
                # ||x_(t-1) - x_(t-1)^new - \delta f(x_(t-1),y)||_2 
                norm_y = torch.linalg.norm(tmp - x_prev.grad)
                norm_y.backward()
                y_grad = measurement.grad # * (tmp - x_prev.grad) / norm_y              
             
                return x_prev_grad, norm, y_grad
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.coef = torch.linspace(10, 100, 1000)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        if kwargs['flag'] == 'forward':
            x_prev_grad, norm, y_grad = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            # norm_grad = torch.clamp(norm_grad, -kwargs["coef2"], kwargs["coef2"])
            """kwargs["coef2"] 是第self.i次去噪的噪声系数, 后一项为调整系数"""

            # x_t -= norm_grad * 1.0 # kwargs["coef2"] * self.coef[self.i]
            """ODE的情况下，不增加噪声导致生成图片含有大量高斯噪声"""

            # tmp = torch.randn_like(x_t) #* kwargs['noise_coef'] #- norm_grad * 0.6# kwargs["coef2"] * self.coef[self.i]
            # if self.i <= 100:
            x_t -= x_prev_grad * 0.6
        
            measurement = measurement - (y_grad / self.i)
            self.i -= 1
            return x_t, norm, measurement
        else: 
            x_prev_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_t -= x_prev_grad * 0.6
            self.i -= 1
            return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
