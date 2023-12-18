# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, modules
import configargparse
import torch

import configargparse
import json
import numpy as np
from torch.utils.data import DataLoader  # noqa: E402
import math
from torch.nn import functional as F

import PIL.Image as Image

command_line = ""
for arg in sys.argv:
    command_line += arg
    command_line += " "

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=True,  help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs_nvp', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False, default="",
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

'''Dataset configure'''
p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g., /data/UVG/Honeybee)")

opt = p.parse_args()

# quantization
unit_multiplier = 2.0**8-1.0

dataset_name = opt.experiment_name

with open(opt.config, 'r') as f:
    config = json.load(f)

# Define the model.
model = modules.CVR(out_features=1, encoding_config=config["cvr"], export_features=False)
model.cuda()

config = config["cvr"]

# Model Loading
root_path = os.path.join(opt.logging_root, opt.experiment_name)

path = os.path.join(root_path, 'checkpoints', "model_best.pth")
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])
# model.set_latent_grid(checkpoint['latent_grid'])

'''Load Volume Dataset'''
volume_dataset = dataio.ImageDataset(path_to_info=opt.dataset, train=False)
dataloader = DataLoader(volume_dataset, shuffle=False, batch_size=1, num_workers=0)

results = {}
results_dir = os.path.join(root_path, 'results')
os.makedirs(results_dir, exist_ok=True)

psnrs = []
model.cuda()
model.eval()

unit_multiplier = 255.0

for step, (model_input, coords, file_name) in enumerate(dataloader):

    with torch.no_grad():

        prediction = model(coords=coords, image=model_input)
        prediction = prediction["model_out"]

        side_length = int(math.sqrt(prediction.shape[1]))
        image = prediction.squeeze().view(side_length, side_length).cpu().numpy()
        # save image to disk as png
        result_name = os.path.join(results_dir, "result_{}".format(file_name[0]))
        image = Image.fromarray(np.uint8(image * unit_multiplier))
        image.save(result_name)

        # export normal interpolated images
        nearest = F.interpolate(model_input, size=[side_length, side_length], mode='nearest')
        bilinear = F.interpolate(model_input, size=[side_length, side_length], mode='bilinear', align_corners=False)

        nearest = nearest.squeeze().cpu().numpy()
        bilinear = bilinear.squeeze().cpu().numpy()

        # save image to disk as png
        nearest_name = os.path.join(results_dir, "nearest_{}".format(file_name[0]))
        image = Image.fromarray(np.uint8(nearest * unit_multiplier))
        image.save(nearest_name)

        linear_name = os.path.join(results_dir, "bilinear_{}".format(file_name[0]))
        image = Image.fromarray(np.uint8(bilinear * unit_multiplier))
        image.save(linear_name)

print("Done!")
    
