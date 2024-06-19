from functools import partial
import os
import argparse
import yaml
import json

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
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name']+"_DCVC_2_quality_elic00016")
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

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    psnr_sum = []
    bpp_list = []
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
       
        # exit 
        if i == 250: 
            exit(0)

        if i == 0:
            ref_frame = None

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(data=ref_img, frame_idx=i, ref_frame=ref_frame, flag=1)
            psnr_sum.append(PSNR(clear_color(ref_img), clear_color(y)))
            
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y))
             
            # y_n = y
            # y_n = noiser(y)

            bpp = operator.getBpp(data=ref_img, frame_idx=i, ref_frame=ref_frame)
            bpp_list.append(bpp)
            ref_frame = y
            continue
        
        # Sampling
        # x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        # sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path, frame_idx=i, ref_frame=ref_frame)
        # ref_frame = ref_img
        # psnr_sum.append(PSNR(clear_color(ref_img), clear_color(sample)))

        # plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        # plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        # plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))


    cur_all_i_frame_quality = 0
    cur_all_p_frame_quality = 0
    cur_all_i_frame_bpp = 0
    cur_all_p_frame_bpp = 0
    frame_num = len(psnr_sum)
    i_frame_num = 0
    for i, psnr in enumerate(psnr_sum):
        if i % 10 == 0:
            cur_all_i_frame_quality += psnr
            cur_all_i_frame_bpp += bpp_list[i]
            i_frame_num += 1
        else:
            cur_all_p_frame_quality += psnr
            cur_all_p_frame_bpp += bpp_list[i]

    results = {'cur_frame_quality': psnr, 'cur_avg_i_frame_psnr': cur_all_i_frame_quality / i_frame_num, 'cur_avg_p_frame_psnr': cur_all_p_frame_quality / (frame_num - i_frame_num),
               'cur_avg_i_frame_bpp': cur_all_i_frame_bpp / i_frame_num, 'cur_avg_p_frame_bpp': cur_all_p_frame_bpp / (frame_num - i_frame_num), 'frame_num': frame_num}
    
    with open(task_config['save_path'], 'w') as f:
        json.dump(results, f, indent=2)
    



if __name__ == '__main__':
    main()
