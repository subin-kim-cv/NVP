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
from ignite.metrics import PSNR


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

path = os.path.join(root_path, 'checkpoints', "model_latest.pth")
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

psnr_metric = PSNR(data_range=1.0)

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
        nearest = nearest.squeeze(0)
        bilinear = bilinear.squeeze(0)

        ## compute PSNR and other metrics of isotropic test data is available 
        if dataset.has_isotropic_test_data():
            # result_psnr = compute_psnr(prediction, gt)

            # print prediction min max
            # print("Min Max prediction", prediction.min(), prediction.max())
            # print("Min Max GT", gt.min(), gt.max())

            psnr_metric.update((prediction, gt))
            result_psnr = psnr_metric.compute()
            psnr_metric.reset()
            result_psnrs.append(result_psnr)

            nearest_tmp = nearest.view(1, side_length * side_length, 1)
            psnr_metric.update((nearest_tmp, gt))
            nearest_psnr = psnr_metric.compute()
            psnr_metric.reset()
            nearest_psnrs.append(nearest_psnr)

            bilinear_tmp = bilinear.view(1, side_length * side_length, 1)
            psnr_metric.update((bilinear_tmp, gt))
            bilinear_psnr = psnr_metric.compute()
            psnr_metric.reset()
            bilinear_psnrs.append(bilinear_psnr)

            print("Result PSNR: {}, Bilinear PSNR: {}, Nearest PSNR: {}".format(result_psnr, bilinear_psnr, nearest_psnr))

        result_name = os.path.join(results_dir, "result_{}".format(file_name[0]))
        nearest_name = os.path.join(results_dir, "nearest_{}".format(file_name[0]))
        linear_name = os.path.join(results_dir, "bilinear_{}".format(file_name[0]))

        folder_name = file_name[0].split(".")[0]
        results_folder = os.path.join(results_dir, folder_name)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        if dataset.has_isotropic_test_data():

            result_psnr_str = str(result_psnr).replace(".", ",")
            result_name = os.path.join(results_folder, "result_psnr_{}_{}".format(file_name[0], result_psnr_str))

            nearest_psnr_str = str(nearest_psnr).replace(".", ",")
            nearest_name = os.path.join(results_folder, "nearest_psnr_{}_{}".format(file_name[0], nearest_psnr_str))

            bilinear_psnr_str = str(bilinear_psnr).replace(".", ",")
            linear_name = os.path.join(results_folder, "bilinear_psnr_{}_{}".format(file_name[0], bilinear_psnr_str))

            gt_name = os.path.join(results_folder, "gt_{}".format(file_name[0]))

        image = prediction.squeeze().view(side_length, side_length).cpu().numpy()
        nearest = nearest.squeeze().cpu().numpy()
        bilinear = bilinear.squeeze().cpu().numpy()
        gt = gt.squeeze().view(bilinear.shape).cpu().numpy()

        image = Image.fromarray(np.uint8(image * unit_multiplier))
        image.save(result_name)

        # save image to disk as png
        image = Image.fromarray(np.uint8(nearest * unit_multiplier))
        image.save(nearest_name)

        image = Image.fromarray(np.uint8(bilinear * unit_multiplier))
        image.save(linear_name)

        image = Image.fromarray(np.uint8(gt * unit_multiplier))
        image.save(gt_name)

print("-------------------------------")
print("Average Result PSNR: {}".format(np.mean(result_psnrs)))
print("Average Bilinear PSNR: {}".format(np.mean(bilinear_psnrs)))
print("Average Nearest PSNR: {}".format(np.mean(nearest_psnrs)))
print("Done!")
    
