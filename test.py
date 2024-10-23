import torch 
import matplotlib.pyplot as plt 
from util.img_utils import clear_color
x = torch.randn((2,6))
y = torch.ones((2,6))
print(x)
x[:,::3] = y[:, ::3]
print(x)