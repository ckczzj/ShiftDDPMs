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

class MNISTShiftPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_class = config["num_class"]
        self.image_channel = config["image_channel"]
        self.image_size = config["image_size"]

        # predict
        # self.predictor = nn.Linear(self.num_class, self.image_channel * self.image_size * self.image_size)

        self.predictor = nn.Sequential(
            nn.Linear(self.num_class, 32),
            Swish(),
            nn.Linear(32, 32 * 8 * 8),
            Swish(),
            View((-1, 32, 8, 8)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 16
            Swish(),
            nn.ConvTranspose2d(32, self.image_channel, 4, 2, 1),  # 32
        )

        # linspace
        # self.mean_matrix = torch.zeros(self.num_class, self.image_channel * self.image_size * self.image_size)
        # for i in range(10):
        #     self.mean_matrix[i] = torch.linspace(-1, 1, 10)[i]
        # self.mean_matrix = self.mean_matrix.cuda()

        # mnist_mean
        # self.mnist_mean = torch.load("./mnist_mean.pt").reshape(self.num_class, self.image_channel * self.image_size * self.image_size)
        # self.mnist_mean = self.mnist_mean.cuda()

    def forward(self, x):
        # predict
        return self.predictor(x).reshape(-1, self.image_channel, self.image_size, self.image_size)

        # linspace
        # return torch.matmul(x, self.mean_matrix).reshape(-1, self.image_channel, self.image_size, self.image_size)

        # mnist_mean
        # return torch.matmul(x, self.mnist_mean).reshape(-1, self.image_channel, self.image_size, self.image_size)