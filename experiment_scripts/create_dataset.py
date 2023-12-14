import argparse
from cloudvolume import CloudVolume
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import math
import random

def main(args):

    # load cloudvolume
    cv = CloudVolume(
        args.path,
        cache=True,
        parallel=True,
    )

    try:
        volume_size = cv.info["scales"][0]["size"]
        resolution = cv.info["scales"][0]["resolution"]
        print("Volume size: ", volume_size)
        print("Resolution: ", resolution)
    except KeyError:
        print("Double check if the neuroglancer precomputed info file contains necessary metadata.")

    output_path = "./data/{}/".format(args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.train:
        output_path = os.path.join(output_path, "train")
    else:
        output_path = os.path.join(output_path, "test")
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    

    # load dataset
    for i in tqdm(range(args.n_images)):
        download_size = [args.image_size, args.image_size, args.image_size]
        # choose a random axis that is not the anistropic axis
        numbers = {0, 1, 2}
        choices = list(numbers - {args.anistropic_dim})
        choice = random.choice(choices)

        if args.train:
            # get random x,y,z coordinates
            x = np.random.randint(args.image_size / 2, volume_size[0] - args.image_size / 2) # assuming x,y is the high resolution axis
            y = np.random.randint(args.image_size / 2, volume_size[1] - args.image_size / 2)
            z = np.random.randint(args.image_size / 2, volume_size[2] - args.image_size / 2)
            
            point = [x, y, z]
            point[args.anistropic_dim] = np.random.randint(0, volume_size[args.anistropic_dim])
            
            download_size[args.anistropic_dim] = 1
        else:
            # get random x,y,z coordinates
            x = np.random.randint(args.image_size / 2, volume_size[0] - args.image_size / 2)
            y = np.random.randint(args.image_size / 2, volume_size[1] - args.image_size / 2)
            z = np.random.randint(args.image_size / 2, volume_size[2] - args.image_size / 2)
            
            point = [x, y, z]
            point[choice] = np.random.randint(0, volume_size[choice]) # randomly pick from the istropic axes

            factor = resolution[args.anistropic_dim] / resolution[choice]
            download_size[args.anistropic_dim] = round(args.image_size / factor)
            download_size[choice] = 1

        # get image
        img = cv.download_point(tuple(point), size=tuple(download_size), mip=0)

        im = Image.fromarray(img.squeeze())
        im.save(os.path.join(output_path, "{}_{}_{}.png".format(x, y, z)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example Script")
    parser.add_argument('--name', type=str, help='Name of dataset', required=True)
    parser.add_argument('--path', type=str, help='Path to ng precomputed file', required=True)
    parser.add_argument('--train', action="store_true", help='Whether to download training or test images')
    parser.add_argument('--anistropic_dim', type=int, help='Which dimension is anistropic. 0-->x, 1-->, 2-->z', default=2, required=True)
    parser.add_argument('--image_size', type=int, help='Pixel size of dataset images', default=256, required=True)
    parser.add_argument('--n_images', type=int, help='Number of images being downloaded', default=1000, required=True)


    args = parser.parse_args()
    main(args)