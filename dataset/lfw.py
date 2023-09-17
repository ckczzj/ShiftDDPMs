import os

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.utils import load_pickle


class my_transform:
    def __init__(self):
        pass

    def __call__(self, x):
        for i in range(3):
            x[i,:,:] *= (np.random.random() * 0.06 + 0.97)
        return x

class LFW(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.image_size = self.config["image_size"]
        self.image_channel = self.config["image_channel"]
        self.data_path = self.config["data_path"]
        self.train = self.config["train"]

        # self.data = torchfile.load(os.path.join(self.data_path,"train.t7" if self.is_training else "test.t7"))
        self.data = load_pickle(os.path.join(self.data_path, "train.pkl" if self.train else "test.pkl"))

        self.images = self.data[b'trainData' if self.train else b'testData']
        self.attributes = self.data[b'trainAttr' if self.train else b'testAttr']
        self.ids = self.data[b'trainId' if self.train else b'testId']

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            my_transform(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ])

        assert len(self.images) == len(self.attributes) == len(self.ids)

        with open(os.path.join(self.data_path,"attributes.txt"), "r") as f:
            lines = f.readlines()
            self.attribute_name = lines[1].strip().split("\t")[3:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 3 x 64 x 64
        image = Image.fromarray(np.transpose(np.clip(self.images[index] * 255, 0, 255).astype(np.uint8), (1, 2, 0)))
        image = self.transform(image)
        gt = np.transpose(np.clip(self.images[index] * 255, 0, 255).astype(np.uint8), (1, 2, 0))
        # 73
        attribute = self.attributes[index].astype(np.float32)

        return {
            "index": index,
            "gt": gt,
            "x_0": image,
            "condition": torch.from_numpy(attribute),
            "caption": str(self.ids[index])
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx = []
        x_0 = []
        gt = []
        condition = []
        captions = []
        for i in range(batch_size):
            idx.append(batch[i]["index"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])
            condition.append(batch[i]["condition"])
            captions.append(batch[i]["caption"])

        x_0 = torch.stack(x_0, dim=0)
        condition = torch.stack(condition, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
            "condition": condition,
            "captions": [s["caption"] for s in batch]
        }
