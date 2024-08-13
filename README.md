# Diffusion Posterior Sampling for Compress
![overview](./figures/lossy%20overview.png)

## Abstract
the work is derived from **Diffusion Posterior Sampling for General Noisy Inverse Problems**, we test different compress method as *Operator* with different compress ratio. If you want to take a test, modify config file and remember *Operator* must have gradient.


## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1


<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/AlbertLin0/diffusionPosteriorSamplingForCompress/tree/submit

cd diffusion-posterior-sampling
```

<br />

### 2) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
From the [link](https://onedrive.live.com/?authkey=%21AGZwZffbRsVcjSQ&id=2866592D5C55DF8C%211198&cid=2866592D5C55DF8C), download the existing video compressor [DCVC](https://github.com/microsoft/DCVC/tree/main/DCVC) and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

:speaker: Checkpoint for imagenet is uploaded.

<br />

### 3) Inference

```
python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={TASK-CONFIG};
```


:speaker: For imagenet, use configs/imagenet_model_config.yaml

<br />


### Structure of task configurations
You need to write your data directory at data.root. Default is ./data/samples which contains three sample images from FFHQ validation set.

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```



