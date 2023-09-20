from PIL import Image

import torch

def numerical_rescale(x, is_0_1, to_0_1):
    if is_0_1 and to_0_1:
        return x.clamp(0., 1.).to(torch.float32)
    elif is_0_1 and not to_0_1:
        return ((x - 0.5) * 2.).clamp(-1., 1.).to(torch.float32)
    elif not is_0_1 and to_0_1:
        return ((x + 1.) / 2.).clamp(0., 1.).to(torch.float32)
    else:
        return x.clamp(-1., 1.).to(torch.float32)

def tensor_to_pillow(x, is_0_1):
    if not is_0_1:
        x = (x + 1.) / 2.
    x = x.mul(255).add(0.5).clamp(0, 255)
    x = x.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(x)


