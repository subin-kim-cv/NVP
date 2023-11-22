import glob
import os
import re

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F

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

    # x_range = torch.linspace(0, 1, width)
    # y_range = torch.linspace(0, 1, height)
    # z_range = torch.linspace(0, 1, depth)
    
    # x, y, z = torch.meshgrid(x_range, y_range, z_range)
    # coords = torch.stack((x, y, z), dim=-1)

    # coords = coords.reshape(-1, 3)
    values = chunk.reshape(-1, channels)

    return coords, values

class VolumeChunk():
    def __init__(self, chunks, sizes, position) -> None:
        super().__init__()

        zipped = sorted(zip(sizes, chunks)) # make sure the is sorted by size in ascending order
        self.sizes, self.chunks = zip(*zipped)
        self.position = position

    def get_random_res_chunk(self):
        # Return random chunk that is at least one level larger than input
        return self.chunks[random.randint(0, len(self.chunks) - 1)]

    def get_sizes(self):
        return self.sizes
    
    def get_res_chunk(self, idx):
        return self.chunks[idx]
    
    def get_max_res_chunk(self):
        return self.chunks[-1]
    
    def get_eval_res_chunk(self):
        high_res = self.chunks[-1].unsqueeze(0).unsqueeze(0)
        eval = F.interpolate(high_res, scale_factor=2, mode='trilinear', align_corners=False)
        return eval.squeeze()
    
    def get_min_res_chunk(self):
        return self.chunks[0]


class VolumeDataset(Dataset):
    def __init__(self, path_to_volume_info, train=True) -> None:
        super().__init__()

        self.info = json.loads(open(path_to_volume_info).read())
        self.directory = os.path.dirname(path_to_volume_info)
        self.n_chunks = self.info["n_chunks"]
        self.sizes = self.info["sizes"]
        self.chunks = []
        self.train = train # True of Train, If false it's a test setting


        for i in range(self.n_chunks[0]):
            for j in range(self.n_chunks[1]):
                for k in range(self.n_chunks[2]):
                    chunk_folder = os.path.join(self.directory, "{}_{}_{}".format(i, j, k))
                    pyramid = []
                    for size in self.sizes:
                        file = os.path.join(chunk_folder, "{}_{}_{}.npy".format(size, size, size))
                        try: 
                            c = torch.from_numpy(np.load(file))
                            pyramid.append(c)
                        except FileNotFoundError:
                            print("File {} not found".format(file))
                            continue
                    chunk = VolumeChunk(pyramid, self.sizes, position=[i, j, k])
                    self.chunks.append(chunk)


        print(self.info)

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        if self.train:
            random_res = chunk.get_random_res_chunk()
            coords, values = linearize(random_res)
            n_input = chunk.get_sizes()[0] ** 2
            indices = torch.randperm(coords.shape[0] - 1)[:n_input]
            coords = coords[indices, :]
            values = values[indices, :]
        else:
            coords, values = linearize(chunk.get_eval_res_chunk())

        input_data = chunk.get_min_res_chunk()

        # normalize data between -1, 1
        input_data = (input_data - 0.5) * 2.0
        # coords = (coords - 0.5) * 2.0
        values = (values - 0.5) * 2.0

        # [model input data, [gt_coords, gt_values]]
        return [input_data, [coords, values]]