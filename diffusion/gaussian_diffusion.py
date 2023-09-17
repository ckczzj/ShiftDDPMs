import math
import torch
import numpy as np

from functools import partial
from tqdm import tqdm

from diffusion.ddim import DDIM


class GaussianDiffusion:
    def __init__(self, config, device):
        super().__init__()
        self.device=device
        self.timesteps = config["timesteps"]
        betas_type = config["betas_type"]
        if betas_type == "linear":
            betas = np.linspace(0.0001, 0.02, self.timesteps)
        elif betas_type == "cosine":
            betas = []
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            max_beta = 0.999
            for i in range(self.timesteps):
                t1 = i / self.timesteps
                t2 = (i + 1) / self.timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            betas = np.array(betas)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.to_torch = to_torch

        self.alphas = to_torch(alphas)
        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = to_torch(posterior_variance)
        # clip the log because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        self.posterior_log_variance_clipped = to_torch(posterior_log_variance_clipped)

        self.x_0_posterior_mean_x_0_coef = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.x_0_posterior_mean_x_t_coef = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.noise_posterior_mean_x_t_coef = to_torch(np.sqrt(1. / alphas))
        self.noise_posterior_mean_noise_coef = to_torch(betas/(np.sqrt(alphas)*np.sqrt(1. - alphas_cumprod)))

        if "shift_type" in config.keys():
            self.shift_type = config["shift_type"]
            if self.shift_type == "prior_shift":
                shift = 1. - np.sqrt(alphas_cumprod)
                # shift = np.array([(i+1)/1000 for i in range(1000)])
                # shift = np.array([((i+1)**2)/1000000 for i in range(1000)])
                # shift = np.array([np.sin((i+1)/timesteps*np.pi/2 - np.pi/2) + 1.0 for i in range(timesteps)])
            elif self.shift_type == "data_normalization":
                shift = - np.sqrt(alphas_cumprod)
            elif self.shift_type == "quadratic_shift":
                shift = np.sqrt(alphas_cumprod) * (1. - np.sqrt(alphas_cumprod))
                # def quadratic(timesteps, t):
                #     return - (1.0 / (timesteps / 2.0) ** 2) * (t - timesteps) * t
                # shift = np.array([quadratic(self.timesteps, i + 1) for i in range(1000)])
            elif self.shift_type == "early":
                shift = np.array([(i + 1) / 600 - 2. / 3. for i in range(1000)])
                shift[:400] = 0
            else:
                raise NotImplementedError

            self.shift = to_torch(shift)
            shift_prev = np.append(0., shift[:-1])
            self.shift_prev = to_torch(shift_prev)
            d = shift_prev - shift / np.sqrt(alphas)
            self.d = to_torch(d)

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    @staticmethod
    def get_ddim_betas_and_timestep_map(ddim_style, original_alphas_cumprod):
        original_timesteps = original_alphas_cumprod.shape[0]
        ddim_step = int(ddim_style[len("ddim"):])
        # data: x_{-1}  noisy latents: x_{0}, x_{1}, x_{2}, ..., x_{T-2}, x_{T-1}
        # encode: treat input x_{-1} as starting point x_{0}
        # sample: treat ending point x_{0} as output x_{-1}
        use_timesteps = set([int(s) for s in list(np.linspace(0, original_timesteps - 1, ddim_step + 1))])
        timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return np.array(new_betas), torch.tensor(timestep_map, dtype=torch.long)

    # x_0: batch_size x channel x height x width
    # t: batch_size
    def q_sample(self, x_0, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def q_posterior_mean(self, x_0, x_t, t):
        shape = x_t.shape
        return self.extract_coef_at_t(self.x_0_posterior_mean_x_0_coef, t, shape) * x_0 \
               + self.extract_coef_at_t(self.x_0_posterior_mean_x_t_coef, t, shape) * x_t

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def noise_p_sample(self, x_t, t, predicted_noise):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise
        log_variance_clipped = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance_clipped).exp() * noise

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def x_0_clip_p_sample(self, x_t, t, predicted_noise, learned_range=None, clip_x_0=True):
        shape = x_t.shape

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
        if clip_x_0:
            predicted_x_0.clamp_(-1,1)
        predicted_mean = self.q_posterior_mean(predicted_x_0, x_t, t)
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def learned_range_to_log_variance(self, learned_range, t):
        shape = learned_range.shape
        min_log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        max_log_variance = self.extract_coef_at_t(torch.log(self.betas), t, shape)
        # The learned_range is [-1, 1] for [min_var, max_var].
        frac = (learned_range + 1) / 2
        return min_log_variance + frac * (max_log_variance - min_log_variance)

    def predicted_noise_to_predicted_x_0(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t \
               - self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

    def predicted_noise_to_predicted_mean(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
               self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise

    def p_loss(self, noise, predicted_noise, weight=None, loss_type="l2"):
        if loss_type == 'l1':
            return (noise - predicted_noise).abs().mean()
        elif loss_type == 'l2':
            if weight is not None:
                return torch.mean(weight * (noise - predicted_noise) ** 2)
            else:
                return torch.mean((noise - predicted_noise) ** 2)
        else:
            raise NotImplementedError()

    """
        regular
    """
    def regular_train_one_batch(self, denoise_fn, x_0, condition=None):
        shape = x_0.shape
        batch_size = shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_fn(x_t, t, condition)

        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def regular_ddpm_sample(self, denoise_fn, x_T, condition=None):
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = denoise_fn(img, t, condition)
            img = self.noise_p_sample(img, t, predicted_noise)
        return img

    def regular_ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_sample_loop(denoise_fn, x_T, condition)

    """
        shift
    """
    def shift_q_sample(self, x_0, u, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.shift, t, shape) * u
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def shift_noise_p_sample(self, x_t, u, t, predicted_noise):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise + \
            self.extract_coef_at_t(self.d, t, shape) * u
        log_variance_clipped = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance_clipped).exp() * noise

    def shift_train_one_batch(self, denoise_fn, shift_predictor, x_0, condition):
        shape = x_0.shape
        t = torch.randint(0, self.timesteps, (shape[0],), device=self.device).long()
        noise = torch.randn_like(x_0)
        u = shift_predictor(condition)
        x_t = self.shift_q_sample(x_0=x_0, u=u, t=t, noise=noise)
        tmp = self.extract_coef_at_t(self.shift, t, shape) * u / self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
        predicted_noise = denoise_fn(x_t, t, None) - tmp
        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def shift_sample(self, denoise_fn, shift_predictor, x_T, condition):
        shape = x_T.shape
        u = shift_predictor(condition)
        if self.shift_type == "prior_shift" or self.shift_type == "early":
            img = x_T + u
        elif self.shift_type == "data_normalization" or self.shift_type == "quadratic_shift":
            img = x_T
        else:
            raise NotImplementedError
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            tmp = self.extract_coef_at_t(self.shift, t, shape) * u / self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = denoise_fn(img, t, None) - tmp
            img = self.shift_noise_p_sample(img, u, t, predicted_noise)
        return img

    def shift_sample_interpolation(self, denoise_fn, x_T, u):
        shape = x_T.shape
        if self.shift_type == "prior_shift" or self.shift_type == "early":
            img = x_T + u
        elif self.shift_type == "data_normalization" or self.shift_type == "quadratic_shift":
            img = x_T
        else:
            raise NotImplementedError
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            tmp = self.extract_coef_at_t(self.shift, t, shape) * u / self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = denoise_fn(img, t, None) - tmp
            img = self.shift_noise_p_sample(img, u, t, predicted_noise)
        return img