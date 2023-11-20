import glob
import os
import re

import random

import numpy as np
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
import json

from utils import make_coord

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of 0 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

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
    
    def __getitem__(self, idx, ):
        chunk = self.chunks[idx]
        
        if self.train:
            random_res = chunk.get_random_res_chunk()
            coords, values = linearize(random_res)
            n_input = chunk.get_sizes()[0] ** 2
            indices = torch.randperm(coords.shape[0] - 1)[:n_input]
            coords = coords[indices, :]
            values = values[indices, :]

        else:
            coords, values = linearize(chunk.get_max_res_chunk())

        input_data = chunk.get_min_res_chunk()

        # normalize data between -1, 1
        input_data = (input_data - 0.5) * 2.0
        # coords = (coords - 0.5) * 2.0
        values = (values - 0.5) * 2.0

        # [model input data, [gt_coords, gt_values]]
        return [input_data, [coords, values]]

class VideoTime(Dataset):
    def __init__(self, path_to_video, split_num=300):
        super().__init__()

        self.split_num = split_num
        print("[i] Video loading start......")
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            print(" videos")
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) 
            if "timelapse" in path_to_video:
                self.vid = self.vid[40:, :, :, :]
            if "GOPR" in path_to_video:
                self.vid = self.vid[:600]
                self.vid = self.vid[0::2]
        else:
            print(" imgs")
            video_path = os.path.join(path_to_video, "*.png")
            files = sorted(glob.glob(video_path), key=sort_key)[:self.split_num]
            tmp_img = Image.open(files[0])
            tmp_img  = np.array(tmp_img)
            tmp_shape = tmp_img.shape

            self.vid = np.zeros((self.split_num, tmp_shape[0], tmp_shape[1], tmp_shape[2]), dtype=np.uint8)
            for idx, f in enumerate(files):
                img = Image.open(f)
                img = np.array(img)
                self.vid[idx] = img
                
        print("[i] Finished")

        self.shape = self.vid.shape[1:-1]
        
        self.nframes = self.vid.shape[0]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid



class VideoTimeWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None):

        self.dataset = dataset
        nframes = self.dataset.nframes
        self.sidelength = sidelength
        
        self.mgrid = get_mgrid(sidelength, dim=2) # [w * h, 3]

        data = torch.from_numpy(self.dataset[0]) 
        self.data = data.view(self.dataset.nframes, -1, self.dataset.channels) # [ f, w * h, 3]

        # batch 
        self.N_samples = 1245184 

        half_dt =  0.5 / nframes

        # modulation input
        self.temporal_steps = torch.linspace(half_dt, 1-half_dt, self.dataset.nframes)
        
        # temporal coords
        self.temporal_coords = torch.linspace(0, 1, nframes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        temporal_coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,)) 
        spatial_coord_idx = torch.randint(0, self.data.shape[1], (self.N_samples,))
        data = self.data[temporal_coord_idx, spatial_coord_idx, :] 
        
        spatial_coords = self.mgrid[spatial_coord_idx, :] 
        temporal_coords = self.temporal_coords[temporal_coord_idx] 
        
        temporal_steps = self.temporal_steps[temporal_coord_idx]

        all_coords = torch.cat((temporal_coords.unsqueeze(1), spatial_coords), dim=1)
            
        in_dict = {'all_coords': all_coords, "temporal_steps": temporal_steps}
        gt_dict = {'img': data}

        return in_dict, gt_dict
