import os
import random

import cv2
from PIL import Image
from io import BytesIO
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from utils.utils import open_lmdb

class RotationTransform():
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

class CELEBAHQ(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = self.config["data_path"]
        self.mask_path = self.config["mask_path"]
        self.image_channel = self.config["image_channel"]
        self.image_size = self.config["image_size"]
        self.train = self.config["train"]

        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])

        self.mask_files = []
        path = os.path.join(self.mask_path, "train_mask" if self.train else "test_mask")
        for _, _, files in os.walk(path):
            for file in files:
                self.mask_files.append(os.path.join(path, file))

    def mask_augment(self, mask_file):
        mask = 255 - cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, int(256 * 0.6), 255, cv2.THRESH_BINARY)

        kernel_size = torch.randint(9, 50, (1,)).item()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel)

        mask = Image.fromarray(mask)

        mask = transforms.Compose([
            RotationTransform(angles=[-90, 0, 90, 180]),
            transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), fill=0),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])(mask)

        # 0-1 matrix, 1 for mask
        # 1 x image_size x image_size
        return (mask > 0.99).float()

    def __len__(self):
        return 30000

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(5)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        # 0-1 matrix, 1 for mask
        # 1 x image_size x image_size
        if self.train:
            selected_mask_file = self.mask_files[torch.randint(0, len(self.mask_files), (1,)).item()]
            selected_mask = self.mask_augment(selected_mask_file)

            while selected_mask.sum() / (selected_mask.shape[1] * selected_mask.shape[2]) <= 0.05:
                selected_mask_path = self.mask_files[torch.randint(0, len(self.mask_files), (1,)).item()]
                selected_mask = self.mask_augment(selected_mask_path)
        else:
            selected_mask_file = self.mask_files[torch.randint(0, len(self.mask_files), (1,)).item()]
            # resize interpolation threshold: 128
            selected_mask = np.array(Image.open(selected_mask_file).resize((self.image_size, self.image_size)))
            selected_mask = torch.from_numpy((selected_mask >= 128).astype(np.uint8)).unsqueeze(0).float()

        masked_gt = (image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255) * ( - selected_mask + 1.)).permute(1,2,0).to('cpu', torch.uint8).numpy()

        return {
            "index": index,
            "gt": gt,
            "x_0": image,
            "mask": selected_mask,  # 1 x image_size x image_size
            "condition": image * ( - selected_mask + 1.),  # 3 x image_size x image_size
            "masked_gt": masked_gt,  # image_size x image_size x image_channel
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx = []
        x_0 = []
        gt = []
        mask = []
        condition = []
        masked_gt = []
        for i in range(batch_size):
            idx.append(batch[i]["index"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])
            mask.append(batch[i]["mask"])
            condition.append(batch[i]["condition"])
            masked_gt.append(batch[i]["masked_gt"])

        x_0 = torch.stack(x_0, dim=0)
        mask = torch.stack(mask, dim=0)
        condition = torch.stack(condition, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
            "mask": mask,
            "condition": condition,
            "masked_gts": np.array(masked_gt),
        }
