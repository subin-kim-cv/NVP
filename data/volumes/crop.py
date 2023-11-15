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
    parser = argparse.ArgumentParser(description="Example script with command-line arguments.")
    parser.add_argument('--path', type=str, help='Path to numpy array to downsample.')
    parser.add_argument('--size', type=str, default="64", help='Side lengths of downsampled data.')

    args = parser.parse_args()
    size = int(args.size)

    print(args)

    data = np.load(args.path)
    cropped = data[:size, :size, :size]

    print("Done. Cropped to size {}".format(cropped.shape))

    out_path = os.path.join(os.path.dirname(args.path), "cropped_%d.npy" % size)
    np.save(out_path, cropped)

if __name__ == "__main__":
    main()