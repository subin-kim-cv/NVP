import numpy as np
import torch
import os

import argparse

def save(data, sizes, path):
    for i, size in enumerate(sizes):
        np.save(os.path.join(path, "resampled_%d.npy" % size), data[i].numpy())
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
    parser.add_argument('--path', type=str, help='Path to numpy array to resample.')
    parser.add_argument('--sizes', type=str, default="3,66,93,109", help='Side lengths of downsampled data.')

    args = parser.parse_args()

    print(args)

    data = torch.from_numpy(np.load(args.path))
    sizes = [int(num) for num in args.sizes.split(',')]
    results = resize(data, sizes)

    save(results, sizes, os.path.dirname(args.path))

if __name__ == "__main__":
    main()