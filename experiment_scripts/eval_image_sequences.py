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


import PIL.Image as Image  # noqa: E402

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

dir = os.path.dirname(opt.dataset)
test_seq_dir = os.path.join(dir, "test_sequence")


# list all folder names in directory
seq_names = os.listdir(test_seq_dir)

results = {}
results_dir = os.path.join(root_path, 'results')
os.makedirs(results_dir, exist_ok=True)

result_psnr_list = []
bilinear_psnr_list = []
nearest_psnr_list = []

for seq in seq_names:

    '''Load Volume Dataset'''
    dataset = dataio.ImageDataset(path_to_info=opt.dataset, train=False, folder=seq)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0)

    seq_res_dir = os.path.join(results_dir, seq)
    if not os.path.exists(seq_res_dir):
        os.makedirs(seq_res_dir)

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

                # print("Result PSNR: {}, Bilinear PSNR: {}, Nearest PSNR: {}".format(result_psnr, bilinear_psnr, nearest_psnr))

            result_name = os.path.join(seq_res_dir, "result_{}".format(file_name[0]))
            nearest_name = os.path.join(seq_res_dir, "nearest_{}".format(file_name[0]))
            linear_name = os.path.join(seq_res_dir, "bilinear_{}".format(file_name[0]))

            i = file_name[0].split(".")[0]
            # results_folder = os.path.join(results_dir, folder_name)

            if dataset.has_isotropic_test_data():

                res_dir = os.path.join(seq_res_dir, "result")
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                result_psnr_str = str(result_psnr).replace(".", ",")
                result_name = os.path.join(res_dir, "result_{}_psnr_{}.png".format(i, result_psnr_str))

                nearest_dir = os.path.join(seq_res_dir, "nearest")
                if not os.path.exists(nearest_dir):
                    os.makedirs(nearest_dir)
                nearest_psnr_str = str(nearest_psnr).replace(".", ",")
                nearest_name = os.path.join(nearest_dir, "nearest_{}_psnr_{}.png".format(i, nearest_psnr_str))

                bilinear_dir = os.path.join(seq_res_dir, "bilinear")
                if not os.path.exists(bilinear_dir):
                    os.makedirs(bilinear_dir)
                bilinear_psnr_str = str(bilinear_psnr).replace(".", ",")
                linear_name = os.path.join(bilinear_dir, "bilinear_{}_psnr_{}.png".format(i, bilinear_psnr_str))


                gt_dir = os.path.join(seq_res_dir, "gt")
                if not os.path.exists(gt_dir):
                    os.makedirs(gt_dir)
                gt_name = os.path.join(gt_dir, "gt_{}".format(file_name[0]))

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

    result_psnr_list.append(np.mean(result_psnrs))
    bilinear_psnr_list.append(np.mean(bilinear_psnrs))
    nearest_psnr_list.append(np.mean(nearest_psnrs))

    print("-------------------------------")
    print("Average Result PSNR: {}".format(np.mean(result_psnrs)))
    print("Average Bilinear PSNR: {}".format(np.mean(bilinear_psnrs)))
    print("Average Nearest PSNR: {}".format(np.mean(nearest_psnrs)))
    print("Done!")

    ## save results to disk
    with open(os.path.join(seq_res_dir, "result_psnrs.txt"), "w") as f:
        f.write("Result PSNR: {}\n".format(np.mean(result_psnrs)))
        f.write("Bilinear PSNR: {}\n".format(np.mean(bilinear_psnrs)))
        f.write("Nearest PSNR: {}\n".format(np.mean(nearest_psnrs)))


# write average results to disk
with open(os.path.join(results_dir, "avg_psnrs.txt"), "w") as f:
    f.write("Result PSNR: {}\n".format(np.mean(result_psnr_list)))
    f.write("Bilinear PSNR: {}\n".format(np.mean(bilinear_psnr_list)))
    f.write("Nearest PSNR: {}\n".format(np.mean(nearest_psnr_list)))
        
