import os

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.utils import load_pickle, get_one_hot


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
            data = load_pickle(os.path.join(self.config["data_path"], name))
            for i in range(10000):
                # print(data[b'labels'][i])
                self.labels.append(data[b'labels'][i])
                self.images.append(data[b'data'][i,].reshape(
                    self.image_channel,
                    self.image_size,
                    self.image_size
                ).transpose(1, 2, 0))

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ])

        self.id2class = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return len(self.images)

    def get_class_name(self, id):
        return self.id2class[id]

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.images[index]))
        label = self.labels[index]
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return {
            "index": index,
            "gt": gt,
            "x_0": image,
            "label": torch.tensor(label),
            "caption": self.get_class_name(label)
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx = []
        x_0 = []
        gt = []
        label = []
        for i in range(batch_size):
            idx.append(batch[i]["index"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])
            label.append(batch[i]["label"])

        x_0 = torch.stack(x_0, dim=0)
        label = torch.stack(label, dim=0)
        condition = get_one_hot(label, 10)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
            "label": label,
            "condition": condition,
            "captions": [s["caption"] for s in batch]
        }
