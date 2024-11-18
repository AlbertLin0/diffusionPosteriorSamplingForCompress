from functools import partial
import os
import argparse
import yaml
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

from torchsummary import summary

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
import numpy as np
def PSNR(input1, input2):
    mse = np.mean((input1 - input2) ** 2)
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    return psnr.item()

def main():
    print(torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    print(model)
    print(" ---------------------- ")
    summary(model)

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config, model=model) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'] + "_ddim_test2_scale" + str(task_config['conditioning']['params']['scale']) + "_step" + str(diffusion_config['timestep_respacing']) + "_x0" + str(task_config['conditioning']['params']['noise_step']))
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((model_config['image_size'], model_config['image_size'])),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # 
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # img_mohu = Image.open('data/test3/test1.png').convert('RGB')
        # img_mohu = transform(img_mohu)
        # img_mohu = img_mohu.to(device).unsqueeze(0)

        bpp = operator.getBpp(ref_img) 
        print(bpp)
        torch.manual_seed(0)
        y = operator.forward(ref_img, mode='forward')
        # print(np.sum(clear_color(img_mohu) - clear_color(y)))
        # print(torch.mean(img_mohu))
        xt = torch.randn_like(ref_img, device = device)
        # xt = sampler.q_sample_loop(img_mohu)
        # print(torch.mean(xt))
        # print(torch.var(xt))
        sample = sampler.p_sample_loop(xt, y, measurement_cond_fn=measurement_cond_fn,truth=ref_img)
        
        file_path_label = os.path.join("./results/elic_vtest/", f"input/test{str(i).zfill(4)}.png")
        file_path = os.path.join("./results/elic_vtest/", f"recon/test28_{str(i).zfill(4)}.png")
        plt.imsave(file_path, clear_color(sample))
        # print(np.sum(clear_color(img_mohu) - clear_color(y)))
        plt.imsave(file_path_label, clear_color(y))


if __name__ == '__main__':
    main()
