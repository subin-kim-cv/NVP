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

from loss_functions import compute_psnr


import PIL.Image as Image

command_line = ""
for arg in sys.argv:
    command_line += arg
    command_line += " "

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=True,  help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
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
dataset = dataio.ImageDataset(path_to_info=opt.dataset, train=False)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0)

results = {}
results_dir = os.path.join(root_path, 'results')
os.makedirs(results_dir, exist_ok=True)

result_psnrs = []
nearest_psnrs = []
bilinear_psnrs = []

model.cuda()
model.eval()

unit_multiplier = 255.0

for step, (model_input, coords, gt, file_name) in enumerate(dataloader):

    with torch.no_grad():

        prediction = model(coords=coords, image=model_input)
        prediction = prediction["model_out"]

        side_length = int(math.sqrt(prediction.shape[1]))

        # export normal interpolated images
        nearest = F.interpolate(model_input, size=[side_length, side_length], mode='nearest')
        bilinear = F.interpolate(model_input, size=[side_length, side_length], mode='bilinear', align_corners=False)

        ## compute PSNR and other metrics of isotropic test data is available 
        if dataset.has_isotropic_test_data():
            result_psnr = compute_psnr(prediction, gt)
            result_psnrs.append(result_psnr)

            nearest_psnr = compute_psnr(nearest.squeeze().view(side_length * side_length, 1), gt)
            nearest_psnrs.append(nearest_psnr)

            bilinear_psnr = compute_psnr(bilinear.squeeze().view(side_length * side_length, 1), gt)
            bilinear_psnrs.append(bilinear_psnr)

            print("Result PSNR: {}, Bilinear PSNR: {}, Nearest PSNR: {}".format(result_psnr, bilinear_psnr, nearest_psnr))

        result_name = os.path.join(results_dir, "result_{}".format(file_name[0]))
        nearest_name = os.path.join(results_dir, "nearest_{}".format(file_name[0]))
        linear_name = os.path.join(results_dir, "bilinear_{}".format(file_name[0]))

        if dataset.has_isotropic_test_data():
            result_psnr_str = str(result_psnr).replace(".", ",")
            result_name = os.path.join(results_dir, "result_psnr_{}_{}".format(result_psnr_str, file_name[0]))

            nearest_psnr_str = str(nearest_psnr).replace(".", ",")
            nearest_name = os.path.join(results_dir, "nearest_psnr_{}_{}".format(nearest_psnr_str, file_name[0]))

            bilinear_psnr_str = str(bilinear_psnr).replace(".", ",")
            linear_name = os.path.join(results_dir, "bilinear_psnr_{}_{}".format(bilinear_psnr_str, file_name[0]))

        image = prediction.squeeze().view(side_length, side_length).cpu().numpy()
        nearest = nearest.squeeze().cpu().numpy()
        bilinear = bilinear.squeeze().cpu().numpy()

        image = Image.fromarray(np.uint8(image * unit_multiplier))
        image.save(result_name)

        # save image to disk as png
        image = Image.fromarray(np.uint8(nearest * unit_multiplier))
        image.save(nearest_name)

        image = Image.fromarray(np.uint8(bilinear * unit_multiplier))
        image.save(linear_name)

print("-------------------------------")
print("Average Result PSNR: {}".format(np.mean(result_psnrs)))
print("Average Bilinear PSNR: {}".format(np.mean(bilinear_psnrs)))
print("Average Nearest PSNR: {}".format(np.mean(nearest_psnrs)))
print("Done!")
    
