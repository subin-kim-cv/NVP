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
p.add_argument('--num_frames', type=int, default=600, required=True, help="Number of video frames to reconstruct")

'''Save configure'''
p.add_argument("--save", default=False, action="store_true", help="Save the frames")

'''Additional properties'''
p.add_argument('--s_interp', type=int, default=-1, required=False, help="Superresolution scale")
p.add_argument('--t_interp', type=int, default=-1, required=False, help="Video frame interpolation scale")

opt = p.parse_args()

# quantization
unit_multiplier = 2.0**8-1.0

dataset_name = opt.experiment_name

with open(opt.config, 'r') as f:
    config = json.load(f)

# Define the model.
model = modules.NVP(out_features=1, encoding_config=config["nvp"])
model.cuda()

config = config["nvp"]

# Model Loading
root_path = os.path.join(opt.logging_root, opt.experiment_name)

path = os.path.join(root_path, 'checkpoints', "model_best.pth")
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])
model.set_latent_grid(checkpoint['latent_grid'])

'''Load Volume Dataset'''
volume_dataset = dataio.VolumeDataset(path_to_volume_info=opt.dataset, train=False)
dataloader = DataLoader(volume_dataset, shuffle=True, batch_size=2, num_workers=0)

results = {}
results_dir = os.path.join(root_path, 'results')
os.makedirs(results_dir, exist_ok=True)

psnrs = []
model.cuda()
model.eval()

for step, (model_input, gt) in enumerate(dataloader):

    with torch.no_grad():

        test_coords = gt[0].cuda()
        prediction = model(coords = test_coords, train=False)

        volume = dataio.volumize(prediction['model_out'], x_dim=128, y_dim=128, z_dim=128)
        volume = (volume - volume.min())/(volume.max()-volume.min())
        volume = volume.squeeze()

        print(prediction["model_out"].shape)
        print("Min Max prediction", prediction['model_out'].min(), prediction['model_out'].max())

        volume = volume.cpu().numpy()
        np.save(os.path.join(results_dir, f"volume_{str(0).zfill(5)}.npy"), volume[0, :, :, :].squeeze())
    
