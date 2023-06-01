import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import load_pickle, get_one_hot


class CIFAR(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config["image_size"]
        self.image_channel = self.config["image_channel"]
        self.data_path = self.config["data_path"]
        self.train = self.config["train"]

        self.images = []
        self.labels = []
        batch_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"] if self.train else ["test_batch"]
        for name in batch_list:
            dict = load_pickle(os.path.join(self.config["data_path"], name))
            for i in range(10000):
                # print(dict[b'labels'][i])
                self.labels.append(dict[b'labels'][i])
                self.images.append(dict[b'data'][i,].reshape(
                    self.image_channel,
                    self.image_size,
                    self.image_size
                ).transpose(1, 2, 0))

        # -1 ~ 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # div 255, 0 ~ 1,
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.id2class = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return len(self.images)

    def get_class_name(self, id):
        return self.id2class[id]

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return {
            "index": index,
            "gt": gt,
            "x_0": image,
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
            "label": torch.tensor(label),
            "caption": self.get_class_name(label)
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        indices = []
        gts = []
        x_0 = []
        x_T = []
        label = []
        for i in range(batch_size):
            indices.append(batch[i]["index"])
            gts.append(batch[i]["gt"])
            x_0.append(batch[i]["x_0"])
            x_T.append(batch[i]["x_T"])
            label.append(batch[i]["label"])

        x_0 = torch.stack(x_0, dim=0)
        x_T = torch.stack(x_T, dim=0)
        label = torch.stack(label, dim=0)
        condition = get_one_hot(label, 10)

        return {
            "net_input": {
                "x_0": x_0,
                "x_T": x_T,
                "condition": condition,
            },
            "gts": np.asarray(gts),
            "label": label,
            "captions": [s["caption"] for s in batch]
        }
