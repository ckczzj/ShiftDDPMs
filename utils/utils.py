import os
import pickle
import random
import textwrap
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import yaml
import json
import lmdb
from PIL import Image


def init_distributed_mode(args):
    args.global_rank = int(os.environ['RANK'])
    args.global_world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

    torch.distributed.init_process_group(backend='nccl')
    torch.distributed.barrier()
    assert torch.distributed.is_initialized()
    print('DDP initialized as global_rank {}/{}, local_rank {}/{}'.format(args.global_rank, args.global_world_size, args.local_rank, args.local_world_size), flush=True)

# base_seed should be large enough to keep 0 and 1 bits balanced
def set_seed(inc, base_seed=666666666):
    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)

# torchvision.transforms may use python RNG
# avoid different workers having same seed
class set_worker_seed_builder():
    def __init__(self, global_rank):
        self.global_rank = global_rank

    def __call__(self, worker_id):
        # use time to avoid same seed when restoring
        # based on seconds, will be almost same for different processes
        base_seed = int(time.time()) % 88888888
        inc = self.global_rank * 128 * 128 + worker_id * 128
        set_seed(inc, base_seed)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def save_yaml(filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f, Dumper=yaml.Dumper)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def open_lmdb(path):
    env = lmdb.open(
        path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    return env.begin(write=False)

# label: B  return: B x N
def get_one_hot(label, N):
    size=list(label.size())
    size.append(N)
    ones=torch.eye(N) # one-hot to be selected
    label=label.view(-1)
    output=ones.index_select(0,label)
    return output.view(*size)

# images: batch_size x image_size x image_size x image_channel
def save_image(images, save_path, captions=None, gts=None, masked_gts=None, num_cols=None):
    figure = plt.figure(figsize=(20, 20))
    num_images = images.shape[0]
    if num_cols is None:
        num_rows = math.ceil(math.sqrt(num_images))
        num_cols = math.ceil(num_images / num_rows)
    else:
        num_rows = math.ceil(num_images / num_cols)

    num_channels = images.shape[-1]
    assert images.shape[1] == images.shape[2]
    image_size = images.shape[1]

    for i in range(num_images):
        axes = plt.subplot(num_rows, num_cols, i + 1)
        axes.axis("off")

        # captions for each subplot
        if captions is not None:
            axes.set_title(textwrap.fill(captions[i], 30))

        # only generated image
        if gts is None:
            if num_channels==1:
                plt.imshow(np.squeeze(images[i],-1).astype('uint8'), cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(images[i].astype('uint8'))

        # generated image, gt image
        elif masked_gts is None:
            if num_channels == 1:
                merge = Image.new('L', (2 * image_size, image_size))
                merge.paste(Image.fromarray(np.squeeze(images[i],-1).astype('uint8'),"L"), (0, 0))
                merge.paste(Image.fromarray(np.squeeze(gts[i],-1).astype('uint8'),"L"), (image_size, 0))
                plt.imshow(np.asarray(merge), cmap='gray', vmin=0, vmax=255)
            else:
                merge = Image.new('RGB', (2 * image_size, image_size))
                merge.paste(Image.fromarray(images[i].astype('uint8')), (0, 0))
                merge.paste(Image.fromarray(gts[i].astype('uint8')), (image_size, 0))
                plt.imshow(np.asarray(merge))

        # generated image, masked image, gt image
        else:
            if num_channels == 1:
                merge = Image.new('L', (3 * image_size, image_size))
                merge.paste(Image.fromarray(np.squeeze(images[i],-1).astype('uint8'),"L"), (0, 0))
                merge.paste(Image.fromarray(np.squeeze(masked_gts[i],-1).astype('uint8'),"L"), (image_size, 0))
                merge.paste(Image.fromarray(np.squeeze(gts[i],-1).astype('uint8'),"L"), (2 * image_size, 0))
                plt.imshow(np.asarray(merge), cmap='gray', vmin=0, vmax=255)
            else:
                merge = Image.new('RGB', (3 * image_size, image_size))
                merge.paste(Image.fromarray(images[i].astype('uint8')), (0, 0))
                merge.paste(Image.fromarray(masked_gts[i].astype('uint8')), (image_size, 0))
                merge.paste(Image.fromarray(gts[i].astype('uint8')), (2 * image_size, 0))
                plt.imshow(np.asarray(merge))

    plt.savefig(save_path)
    print('save figure to {}'.format(save_path))
    plt.close()
    return figure

def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.cuda(device=torch.cuda.current_device(), non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(1).mean(1).mean(1)


def calculate_ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)

def calculate_lpips(img1, img2, lpips_fn):
    return lpips_fn.forward(img1, img2).view(-1)

def calculate_mse(img1, img2):
    return (img1 - img2).pow(2).mean(dim=[1, 2, 3])