import glob
import os
import re

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
import math
from PIL import Image
import torchvision.transforms as transforms

from utils import make_coord

def sort_key(path):
    # Extract numbers and convert them to integers
    numbers = re.findall(r'\d+', path)
    return [int(num) for num in numbers]

def volumize(chunk, x_dim, y_dim, z_dim):
    chunk = chunk.reshape(chunk.shape[0], x_dim, y_dim, z_dim)
    return chunk

def linearize(chunk):
    chunk = chunk.unsqueeze(0)
    channels, width, height, depth = chunk.shape

    coords = make_coord((width, height, depth)).cuda()
    values = chunk.reshape(-1, channels)

    return coords, values

class VolumeChunk():
    def __init__(self, chunk, position) -> None:
        super().__init__()
        self.data = chunk
        self.position = position


    def get_random_crop(self, x_dim, y_dim, z_dim):
        # Return random crop of chunk with given dimensions
        x, y, z = self.data.shape
        x_start = random.randint(0, x - x_dim)
        y_start = random.randint(0, y - y_dim)
        z_start = random.randint(0, z - z_dim)

        return self.data[x_start:x_start + x_dim, y_start:y_start + y_dim, z_start:z_start + z_dim]

    def get_chunk(self):
        return self.data 
    
    def get_position(self):
        return self.position

class VolumeDataset(Dataset):
    def __init__(self, path_to_volume_info, train=True) -> None:
        super().__init__()

        self.info = json.loads(open(path_to_volume_info).read())
        self.directory = os.path.dirname(path_to_volume_info)
        self.n_chunks = self.info["n_chunks"]
        self.chunks = []
        self.train = train # True of Train, If false it's a test setting
        self.base_resolution = self.info["model_input_size"]

        data_dir = os.path.join(os.path.dirname(path_to_volume_info), "chunks")

        for name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, name)
            data = torch.from_numpy(np.load(file_path))
            
            parts = name.replace(".npy", "").split('_')
            pos = [int(part) for part in parts]

            print("Loading chunk at position {}".format(pos))

            chunk = VolumeChunk(data, position=pos)
            self.chunks.append(chunk)

        print(self.info)

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        scale = random.uniform(1, 3)
        b_res = self.base_resolution
        gt_chunk_res = math.floor(scale * self.base_resolution)

        if self.train:
            crop = chunk.get_random_crop(gt_chunk_res, gt_chunk_res, gt_chunk_res)
            input_data = F.interpolate(crop.unsqueeze(0).unsqueeze(0), size=[b_res, b_res, b_res], mode='trilinear', align_corners=False)
            input_data = input_data.squeeze(1)

        else:
            crop = chunk.get_chunk()
            input_data = crop.unsqueeze(0)

        gt_coords, gt_values = linearize(crop)

        if self.train:
            # sample n random points from gt data
            n_samples = self.base_resolution ** 3
            indices = torch.randperm(gt_coords.shape[0])[:n_samples]
            coords = gt_coords[indices, :]
            values = gt_values[indices, :]
        else:
            t_size = crop.squeeze().shape
            factor = 3
            coords = make_coord((t_size[0] * factor, t_size[1] * factor, t_size[2] * factor)).cuda()
            values = torch.zeros_like(coords[:, 0]).unsqueeze(1).cuda()
            
        # normalize data between -1, 1
        input_data = (input_data - 0.5) * 2.0
        # coords = (coords - 0.5) * 2.0
        values = (values - 0.5) * 2.0

        positions = torch.tensor(chunk.get_position()).squeeze()
        # [model input data, [gt_coords, gt_values]]
        return [input_data, [coords, values], positions]
    

class ImageDataset(Dataset):
    def __init__(self, path_to_info, train=True) -> None:
        super().__init__()
        info = json.loads(open(path_to_info).read())
        self.is_train = train
        self.n_images = info["n_images"]
        self.path = os.path.join(os.path.dirname(path_to_info), "train" if self.is_train else "test")
        self.files = os.listdir(self.path)
        self.anisotropic_factor = 8  # TODO make dynamic

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        file_name = self.files[idx]
        path = os.path.join(self.path, file_name)

        image = Image.open(path) # Load the image
        transform = transforms.ToTensor() # Transform to tensor
        gt = transform(image).cuda() # Transform to tensor, already in [0, 1]
    
        batched_gt = gt.unsqueeze(0)
        anisotropic_shape = [batched_gt.shape[-2], int(batched_gt.shape[-1] / self.anisotropic_factor)]
        input = F.interpolate(batched_gt, size=anisotropic_shape, mode='nearest')
        input = input.squeeze(0)

        coords = make_coord(batched_gt.shape[-2:]).cuda()

        gt = gt.permute(1, 2, 0)
        gt = gt.view(-1, gt.shape[-1])

        return [input, coords, gt]

