import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import setting


class dataset(Dataset):
    def __init__(
        self,
        batch_size,
        input_image_paths,
        raw_image_paths,
        img_size=512,
        raw_path=setting.raw_path,
        input_path=setting.input_path,
    ):
        self.batch_size = batch_size
        self.input_path = input_path
        self.raw_path = raw_path
        self.img_size = img_size
        self.input_image_paths = input_image_paths
        self.raw_image_paths = raw_image_paths
        self.len = len(self.input_image_paths) // batch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.batch_input_img = [
            self.input_image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
            for idx in range(self.len)
        ]

        self.batch_raw_img = [
            self.raw_image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
            for idx in range(self.len)
        ]

    def __getitem__(self, idx):
        input = torch.stack(
            [
                self.transform(Image.open(os.path.join(self.input_path, file_name)))
                for file_name in self.batch_input_img[idx]
            ]
        )
        raw = torch.stack(
            [
                self.transform(Image.open(os.path.join(self.raw_path, file_name)))
                for file_name in self.batch_raw_img[idx]
            ]
        )

        return input, raw

    def __len__(self):
        return self.len
