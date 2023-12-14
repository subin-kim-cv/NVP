import argparse
from cloudvolume import CloudVolume
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

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

    # load dataset
    for i in tqdm(range(args.n_images)):
        # get random x,y,z coordinates
        x = np.random.randint(args.image_size / 2, volume_size[0] - args.image_size / 2) # assuming x,y is the high resolution axis
        y = np.random.randint(args.image_size / 2, volume_size[1] - args.image_size / 2)
        z = np.random.randint(0, volume_size[2])

        # get image
        img = cv.download_point( (x, y, z), size=(args.image_size, args.image_size, 1), mip=0)
        im = Image.fromarray(img.squeeze())
        im.save(os.path.join(output_path, "{}_{}_{}.png".format(x, y, z)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example Script")
    parser.add_argument('--name', type=str, help='Name of dataset', required=True)
    parser.add_argument('--path', type=str, help='Path to ng precomputed file', required=True)
    parser.add_argument('--volume_size', type=str, help='X,Y,Z dimensions of volume in voxels', required=False)
    parser.add_argument('--image_size', type=int, help='Pixel size of dataset images', default=256, required=False)
    parser.add_argument('--n_images', type=int, help='Number of images being downloaded', default=1000, required=False)
    args = parser.parse_args()
    main(args)