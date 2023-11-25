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
        scale = random.uniform(1, 4)

        b_res = self.base_resolution

        gt_chunk_res = math.floor(scale * self.base_resolution)


        crop = chunk.get_random_crop(gt_chunk_res, gt_chunk_res, gt_chunk_res)
        if not self.train:
            crop = chunk.get_chunk()

        # downsample crop to base resolution
        input_data = F.interpolate(crop.unsqueeze(0).unsqueeze(0), size=[b_res, b_res, b_res], mode='trilinear', align_corners=False)
        input_data = input_data.squeeze(1)

        # if self.train:
        # generate gt data
        gt_coords, gt_values = linearize(crop)

        # sample n random points from gt data
        n_samples = self.base_resolution ** 3
        indices = torch.randperm(gt_coords.shape[0])[:n_samples]
        coords = gt_coords[indices, :]
        values = gt_values[indices, :]

        if not self.train:
            coords = gt_coords
            values = gt_values
            
        # normalize data between -1, 1
        input_data = (input_data - 0.5) * 2.0
        # coords = (coords - 0.5) * 2.0
        values = (values - 0.5) * 2.0

        positions = torch.tensor(chunk.get_position()).squeeze()
        # [model input data, [gt_coords, gt_values]]
        return [input_data, [coords, values], positions]