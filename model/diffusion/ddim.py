import torch
import numpy as np

from functools import partial
from tqdm import tqdm

class DDIM:
    def __init__(self, betas, timestep_map, device):
        super().__init__()
        self.device=device
        self.timestep_map = timestep_map.to(self.device)
        self.timesteps = betas.shape[0] - 1

        # length = timesteps + 1
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 1. will never be used
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)  # 0. will never be used

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        # self.alphas = to_torch(alphas)
        # self.betas = to_torch(betas)
        # self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)


        # self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        # self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    def t_transform(self, t):
        new_t = self.timestep_map[t]
        return new_t

    def ddim_sample(self, denoise_fn, x_t, t, condition):
        shape = x_t.shape
        predicted_noise = denoise_fn(x_t, self.t_transform(t), condition)
        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

    def ddim_sample_loop(self, denoise_fn, x_T, condition):
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.ddim_sample(denoise_fn, img, t, condition)
        return img

    # def ddim_encode(self, denoise_fn, x_t, t, condition):
    #     shape = x_t.shape
    #     predicted_noise = denoise_fn(x_t, self.t_transform(t), condition)
    #     predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
    #                     self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
    #     predicted_x_0 = predicted_x_0.clamp(-1, 1)
    #
    #     new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
    #                           self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)
    #
    #     alpha_bar_next = self.extract_coef_at_t(self.alphas_cumprod_next, t, shape)
    #
    #
    #     return predicted_x_0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * new_predicted_noise
    #
    # def ddim_encode_loop(self, denoise_fn, x_0, condition):
    #     shape = x_0.shape
    #     batch_size = shape[0]
    #     x_t = x_0
    #     for i in tqdm(range(0, self.timesteps), desc='encoding loop time step', total=self.timesteps):
    #         t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
    #         x_t = self.ddim_encode(denoise_fn, x_t, t, condition)
    #     return x_t
