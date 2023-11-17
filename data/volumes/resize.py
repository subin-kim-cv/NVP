import numpy as np
import torch
import os

import json

import argparse

def save(data, sizes, path):
    for i, size in enumerate(sizes):
        np.save(os.path.join(path, "{}_{}_{}.npy".format(size, size, size)), data[i].numpy())
    print("Done. Saved resampled data to {}".format(path))


def resize(data, sizes):
    output = []
    data = data.unsqueeze(0).unsqueeze(0)
    for size in sizes:
        resampled = torch.nn.functional.interpolate(data, size=[size, size, size], mode='trilinear', align_corners=False)
        resampled = resampled.squeeze()
        print("Downsampled array to {}".format(resampled.shape))
        output.append(resampled)
    return output

def main():
    parser = argparse.ArgumentParser(description="Resampled a given (3D) numpy array using trilinear interpolation.")
    parser.add_argument('--path', type=str, help='Path to info.json file of dataset.')
    parser.add_argument('--sizes', type=str, default="3,66,93,109", help='Side lengths of downsampled data.')

    args = parser.parse_args()

    print(args)

    info = json.loads(open(args.path).read())
    root = os.path.dirname(args.path)
    n_chunks = info["n_chunks"]
    x_chunk_size = info["chunk_size"][0]
    y_chunk_size = info["chunk_size"][1]
    z_chunk_size = info["chunk_size"][2]

    sizes = [int(num) for num in args.sizes.split(',')]

    for x in range(n_chunks[0]):
        for y in range(n_chunks[1]):
            for z in range(n_chunks[2]):
                chunk_folder = os.path.join(root, "{}_{}_{}".format(x, y, z))
                data = np.load(os.path.join(chunk_folder, "{}_{}_{}.npy".format(x_chunk_size, y_chunk_size, z_chunk_size)))
                data = torch.from_numpy(data)
                results = resize(data, sizes)
                save(results, sizes, chunk_folder)
    
    print("Done. Saved resampled data to chunk folders in {}".format(root))

if __name__ == "__main__":
    main()