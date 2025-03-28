import torch
import math
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plot
import cv2


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# 时间步(timestep)定义为1000
timesteps = 1000

# 定义Beta Schedule, 选择线性版本，同DDPM原文一致，当然也可以换成cosine_beta_schedule
betas = linear_beta_schedule(timesteps=timesteps)

# 根据beta定义alpha
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# 计算前向过程 diffusion q(x_t | x_{t-1}) 中所需的
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# 前向加噪过程: forward diffusion process
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
        cv2.imwrite('noise.png', noise.numpy() * 255)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    print('sqrt_alphas_cumprod_t :', sqrt_alphas_cumprod_t)
    print('sqrt_one_minus_alphas_cumprod_t :', sqrt_one_minus_alphas_cumprod_t)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# 图像后处理
def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = x_noisy.squeeze().numpy()

    return noisy_image


...

# 展示图像, t=0, 50, 100, 500的效果
x_start = cv2.imread('img.png') / 255.0
x_start = torch.tensor(x_start, dtype=torch.float)
cv2.imwrite('img_0.png', get_noisy_image(x_start, torch.tensor([0])) * 255.0)
cv2.imwrite('img_50.png', get_noisy_image(x_start, torch.tensor([50])) * 255.0)
cv2.imwrite('img_100.png', get_noisy_image(x_start, torch.tensor([100])) * 255.0)
cv2.imwrite('img_500.png', get_noisy_image(x_start, torch.tensor([500])) * 255.0)
cv2.imwrite('img_999.png', get_noisy_image(x_start, torch.tensor([999])) * 255.0)

sqrt_alphas_cumprod_t: tensor([[[0.9999]]], dtype=torch.float64)
sqrt_one_minus_alphas_cumprod_t: tensor([[[0.0100]]], dtype=torch.float64)
sqrt_alphas_cumprod_t: tensor([[[0.9849]]], dtype=torch.float64)
sqrt_one_minus_alphas_cumprod_t: tensor([[[0.1733]]], dtype=torch.float64)
sqrt_alphas_cumprod_t: tensor([[[0.9461]]], dtype=torch.float64)
sqrt_one_minus_alphas_cumprod_t: tensor([[[0.3238]]], dtype=torch.float64)
sqrt_alphas_cumprod_t: tensor([[[0.2789]]], dtype=torch.float64)
sqrt_one_minus_alphas_cumprod_t: tensor([[[0.9603]]], dtype=torch.float64)
sqrt_alphas_cumprod_t: tensor([[[0.0064]]], dtype=torch.float64)
sqrt_one_minus_alphas_cumprod_t: tensor([[[1.0000]]], dtype=torch.float64)
