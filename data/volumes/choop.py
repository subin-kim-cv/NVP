import numpy as np
import os
import json

import argparse

def main():
    
    parser = argparse.ArgumentParser(description="This script chopps a 3D numpy array into n chunks per dimension.")
    parser.add_argument('--path', type=str, help='Path to info.json file of dataset.')
    # parser.add_argument('--n_chunk', type=str, default="2,2,2", help='Side lengths of downsampled data.')

    args = parser.parse_args()
    print(args)
    
    info = json.loads(open(args.path).read())
    gt_size = info["gt_size"]

    n_chunks = info["n_chunks"]

    data_path = os.path.join(os.path.dirname(args.path), "gt", "{}_{}_{}.npy".format(gt_size[0], gt_size[1], gt_size[2]))
    data = np.load(data_path)
    data = data.astype(np.float32)
    # normalize to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    if len(data.shape) != 3:
        raise ValueError("Data must be 3D, but is {}".format(data.shape))
    if data.shape[0] % n_chunks[0] or  data.shape[1] % n_chunks[1] or data.shape[2] % n_chunks[2]!= 0:
        raise ValueError("Data shape {} is not divisible by n_chunks {}. Please pick n_chunks s.t. it divides the shape of the array".format(data.shape, n_chunks))

    for x in range(n_chunks[0]):
        for y in range(n_chunks[1]):
            for z in range(n_chunks[2]):
                chunk = data[
                    x * data.shape[0] // n_chunks[0] : (x + 1) * data.shape[0] // n_chunks[0],
                    y * data.shape[1] // n_chunks[1] : (y + 1) * data.shape[1] // n_chunks[1],
                    z * data.shape[2] // n_chunks[2] : (z + 1) * data.shape[2] // n_chunks[2],
                ]
                print("Chunk shape: {}".format(chunk.shape))

                directory = os.path.join(os.path.dirname(args.path), "chunks")
                np.save(
                    os.path.join(
                        directory,
                        "{}_{}_{}.npy".format(x, y, z),
                    ),
                    chunk,
                )

    print("Done. Chopped volume into [{}, {}, {}] chunks of size {}".format(n_chunks[0], n_chunks[1], n_chunks[2], chunk.shape))

if __name__ == "__main__":
    main()