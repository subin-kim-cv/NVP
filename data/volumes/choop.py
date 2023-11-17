import numpy as np
import os

import argparse

def main():
    
    parser = argparse.ArgumentParser(description="This script chopps a 3D numpy array into n chunks per dimension.")
    parser.add_argument('--path', type=str, help='Path to numpy array to downsample.')
    parser.add_argument('--n_chunk', type=str, default="2,2,2", help='Side lengths of downsampled data.')

    args = parser.parse_args()
    print(args)
    
    n_chunks = [int(num) for num in args.n_chunk.split(',')]
    data = np.load(args.path)

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

                directory = os.path.join(os.path.dirname(args.path), "{}_{}_{}".format(x, y, z))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                np.save(
                    os.path.join(
                        directory,
                        "{}_{}_{}.npy".format(chunk.shape[0], chunk.shape[1], chunk.shape[2]),
                    ),
                    chunk,
                )

    print("Done. Chopped volume into [{}, {}, {}] chunks of size {}".format(n_chunks[0], n_chunks[1], n_chunks[2], chunk.shape))

if __name__ == "__main__":
    main()