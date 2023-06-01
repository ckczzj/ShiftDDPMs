import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class LFWShiftPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_class = config["num_class"]
        self.image_channel = config["image_channel"]
        self.image_size = config["image_size"]

        # self.predictor = nn.Linear(self.num_class, self.image_channel * self.image_size * self.image_size)

        self.predictor = nn.Sequential(
            nn.Linear(self.num_class, 256),
            Swish(),
            nn.Linear(256, 256 * 2 * 2),
            Swish(),
            View((-1, 256, 2, 2)),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # 4
            Swish(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8
            Swish(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 16
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32
            Swish(),
            nn.ConvTranspose2d(64, self.image_channel, 4, 2, 1),  # 64
        )

    def forward(self, x, **kwargs):
        # return self.predictor(x).reshape(-1, self.image_channel, self.image_size, self.image_size)
        return self.predictor(x)