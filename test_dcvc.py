import torch 
import yaml
import os
from torchvision import transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from guided_diffusion.measurements import get_noise, get_operator
from util.logger import get_logger
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0)*255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)

def main():
    task_config = load_yaml('configs/elic_config.yaml')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # logger
    logger = get_logger()

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    out_path = 'result_DCVC'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    with torch.no_grad():
        for i, ref_img in enumerate(loader):
            logger.info(f"Inference for image {i}")
            fname = str(i).zfill(5) + '.png'
            ref_img = ref_img.to(device)
            if i == 0:
                ref_frame = None
            y = operator.forward(data=ref_img, frame_idx=i, ref_frame=ref_frame)
            ref_frame = ref_img
            plt.imsave(os.path.join(out_path,  fname), clear_color(y))
            # write_torch_frame(y.squeeze(), os.path.join(out_path,  fname))

if __name__ == '__main__':
    main()