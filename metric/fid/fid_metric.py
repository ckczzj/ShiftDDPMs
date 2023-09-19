import os

from PIL import Image
import numpy as np
from scipy import linalg

import torch
from torch.nn.functional import adaptive_avg_pool2d

from metric.fid.inception import InceptionV3

def numerical_rescale(x, normalize_input):
    if normalize_input:
        res = (x + 1.) / 2.
    else:
        res = x
    return res.clamp(0., 1.).to(torch.float32)

def tensor_to_pillow(x, normalize_input):
    if normalize_input:
        x = (x + 1.) / 2.
    x = x.mul(255).add(0.5).clamp(0, 255)
    x = x.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(x)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on a
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on a
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# process image tensor (b x c x h x w) in range (-1., 1.)
class FIDMetric:
    def __init__(self, dims, inception_path, device, target_path = None, img_save_path = None):
        super().__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx], normalize_input=False, inception_path=inception_path).to(device)
        self.inception.eval()
        self.device = device
        self.img_save_path = img_save_path
        self.target_path = target_path
        self.results = []

    def save_images(self, images, image_ids, normalize_input):
        print("Saving to target path.")
        for idx, image in enumerate(images):
            pil_img = tensor_to_pillow(image, normalize_input)
            sub_idx = 0
            while os.path.exists(f'{self.img_save_path}/image_{image_ids[idx]}_{sub_idx}.png'):
                sub_idx += 1
            pil_img.save(f'{self.img_save_path}/image_{image_ids[idx]}_{sub_idx}.png')

    # b x c x h x w
    def process(self, samples, image_ids=None, normalize_input=False):
        if self.img_save_path is not None:
            assert image_ids is not None, "image_ids must be provided to save images."
            self.save_images(samples, image_ids, normalize_input)

        samples = numerical_rescale(samples, normalize_input)
        with torch.no_grad():
            # b x 2048 x 1 x 1
            prediction = self.inception(samples)[0]
        if prediction.size(2) != 1 or prediction.size(3) != 1:
            prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
        # b x 2048
        prediction = prediction.flatten(1, 3).cpu().numpy()
        self.results.append(prediction)

    def all_gather_results(self, world_size):
        gather_results_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gather_results_list, self.results)
        gather_results = []
        for results in gather_results_list:
            gather_results.extend(results)
        return gather_results

    def compute_metrics(self, results):
        # n x 2048
        predictions = np.concatenate(results, axis=0)
        print(predictions.shape)
        mu_prediction, sigma_prediction = np.mean(predictions, axis=0), np.cov(predictions, rowvar=False)
        targets = torch.load(self.target_path)
        mu_target, sigma_target = targets['mu'], targets['sigma']
        fid = calculate_frechet_distance(mu_prediction, sigma_prediction, mu_target, sigma_target)
        return fid

    def compute_stats(self, results):
        # n x 2048
        predictions = np.concatenate(results, axis=0)
        print(predictions.shape)
        mu, sigma = np.mean(predictions, axis=0), np.cov(predictions, rowvar=False)
        return mu, sigma # mu: np.float32, sigma: np.float64

    def reset(self):
        self.results = []
