import os

from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
from utils import load_pickle


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
            transforms.ToTensor(),  # 0 ~ 1
            transforms.RandomHorizontalFlip(),
            my_transform(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),  # -1 ~ 1
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
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
            "condition": torch.from_numpy(attribute),
            "caption": str(self.ids[index])
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        indices = []
        gts = []
        x_0 = []
        x_T = []
        condition = []
        captions = []
        for i in range(batch_size):
            indices.append(batch[i]["index"])
            gts.append(batch[i]["gt"])
            x_0.append(batch[i]["x_0"])
            x_T.append(batch[i]["x_T"])
            condition.append(batch[i]["condition"])
            captions.append(batch[i]["caption"])

        x_0 = torch.stack(x_0, dim=0)
        x_T = torch.stack(x_T, dim=0)
        condition = torch.stack(condition, dim=0)

        return {
            "net_input": {
                "x_0": x_0,
                "x_T": x_T,
                "condition": condition,
            },
            "gts": np.asarray(gts),
            "captions": [s["caption"] for s in batch]
        }
