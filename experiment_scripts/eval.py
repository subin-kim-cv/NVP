# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, modules
import configargparse
import torch
import time
from tqdm import tqdm
from torch import nn
import math

import configargparse
import json
from torchvision.utils import save_image


def quantize_keyframes(params, config):
    params_in_level = []
    resolution_in_level = []
    total = 0
    n_levels = config["n_levels"]
    dim = config["n_features_per_level"]
    per_level_scale = config["per_level_scale"]

    # params in level
    for i in range(n_levels):
        a = math.exp(i * (math.log(per_level_scale)))*16-1
        b = int(math.ceil(a)+1)
        resolution_in_level.append(b)
        c = b**2
        params_in_level.append(total)
        total += c
    params_in_level.append(total) 

    keyframe_features_list = []
    keyframe_features_max_list = []
    keyframe_features_min_list = []

    keyframe_features = params.reshape(-1, dim)
    for d in range(dim):
        keyframe_features_list.append(keyframe_features[:, d])
    
    check = keyframe_features_list[0].shape[0]*dim

    for d in range(dim):

        keyframe_features_max_list.append([])
        keyframe_features_min_list.append([])
        for i in range(n_levels):
            if i != n_levels-1:
                pos1 = params_in_level[i]
                pos2 = params_in_level[i+1]
                tmp_features = keyframe_features[:, d][pos1:pos2]
            else:
                pos1 = params_in_level[i]
                tmp_features = keyframe_features[:, d][pos1:]
            
            check -= len(tmp_features)
            
            keyframe_features_max_list[d].append(torch.max(tmp_features))
            keyframe_features_min_list[d].append(torch.min(tmp_features))
    

            # quantize to 8bit 
            quantized_tmp_features = (tmp_features-keyframe_features_min_list[d][i])/(keyframe_features_max_list[d][i]-keyframe_features_min_list[d][i])
            quantized_tmp_features = unit_multiplier*quantized_tmp_features
            quantized_tmp_features = torch.tensor(quantized_tmp_features+0.5, dtype=torch.uint8)
            quantized_tmp_features = torch.clamp(quantized_tmp_features, 0, 255)

            # de-quantize to 32bit 
            dequantized_tmp_features = torch.tensor(quantized_tmp_features, dtype=torch.float32)
            dequantized_tmp_features = dequantized_tmp_features/unit_multiplier
            dequantized_tmp_features = (keyframe_features_max_list[d][i]-keyframe_features_min_list[d][i])*dequantized_tmp_features+keyframe_features_min_list[d][i]

            if i != n_levels-1:
                keyframe_features[:, d][pos1:pos2] = dequantized_tmp_features
            else:
                keyframe_features[:, d][pos1:] = dequantized_tmp_features

    return nn.Parameter(keyframe_features.reshape(-1).cuda())

def quantize_sparse_grid(params, config):
    dim = config["n_features_per_level"]
    sparse_features = params.clone().detach() # [t, h, w, d]
    sparse_features_min = []
    sparse_features_max = []

    for d in range(dim):

        tmp_sparse_features = sparse_features[:, :, :, d]
        sparse_features_min.append(torch.min(tmp_sparse_features))
        sparse_features_max.append(torch.max(tmp_sparse_features))

        # quantize to 8 bit 
        quantized_tmp_features = (tmp_sparse_features-sparse_features_min[d])/(sparse_features_max[d]-sparse_features_min[d])
        quantized_tmp_features = unit_multiplier*quantized_tmp_features
        quantized_tmp_features = torch.tensor(quantized_tmp_features+0.5, dtype=torch.uint8)
        quantized_tmp_features = torch.clamp(quantized_tmp_features, 0, 255)
        
        # de-quantize to 32bit
        dequantized_tmp_features = torch.tensor(quantized_tmp_features, dtype=torch.float32)
        dequantized_tmp_features = dequantized_tmp_features/unit_multiplier
        dequantized_tmp_features = (sparse_features_max[d]-sparse_features_min[d])*dequantized_tmp_features+sparse_features_min[d]

        sparse_features[:, :, :, d] = dequantized_tmp_features

    return nn.Parameter(sparse_features.cuda())

#############################################################################################################


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
model = modules.NVP(type='nvp', in_features=2, out_features=3, encoding_config=config["nvp"])
model.cuda()

config = config["nvp"]

# Model Loading
root_path = os.path.join(opt.logging_root, opt.experiment_name)

path = os.path.join(root_path, 'checkpoints', "model_best.pth")
checkpoint = torch.load(path)        
print("Epoch: ", checkpoint["epoch"])
# assert checkpoint["epoch"] <= 7676
model.load_state_dict(checkpoint['model'])

'''Save trained parameters'''
keyframe_xy_saved = torch.tensor(model.keyframes_xy.params.clone().detach()).cpu()
keyframe_xt_saved = torch.tensor(model.keyframes_xt.params.clone().detach()).cpu()
keyframe_yt_saved = torch.tensor(model.keyframes_yt.params.clone().detach()).cpu()
sparse_grid_saved = torch.tensor(model.sparse_grid.embeddings.clone().detach()).cpu()


'''Load Quantized Learnable Keyframes'''
model.keyframes_xy.params = quantize_keyframes(keyframe_xy_saved, config["2d_encoding_xy"])
model.keyframes_xt.params = quantize_keyframes(keyframe_xt_saved, config["2d_encoding_xt"])
model.keyframes_yt.params = quantize_keyframes(keyframe_yt_saved, config["2d_encoding_yt"])

print("[i] Loading learnable keyframes finished")



'''Load Quantized Sparse Grid'''
model.sparse_grid.embeddings = quantize_sparse_grid(sparse_grid_saved, config["3d_encoding"])

print("[i] Loading sparse grid finished")


'''Load Video Dataset'''
video_path = opt.dataset
vid_dataset = dataio.VideoTime(video_path, split_num=opt.num_frames)

''' BPP Calculation '''
cur_bpp = utils.get_n_params(model)*32/(vid_dataset.vid.shape[0]*vid_dataset.vid.shape[1]*vid_dataset.vid.shape[2])
quantized_bpp = (utils.get_n_params(model)*8+utils.get_n_params(model.wrapper)*24)/(vid_dataset.vid.shape[0]*vid_dataset.vid.shape[1]*vid_dataset.vid.shape[2])

results = {}
results_dir = os.path.join(root_path, 'results', "quantize")

# Get ground truth and input data
gt_frames = vid_dataset.vid
nframes = vid_dataset.nframes
org_nframes = nframes
resolution = vid_dataset.shape

temporal_interp=False
if opt.s_interp != -1:
    resolution = (resolution[0]*opt.s_interp, resolution[1]*opt.s_interp)
    results_dir += f"_s_interp_{opt.s_interp}"
    quantized_bpp = quantized_bpp/(opt.s_interp*opt.s_interp)
if opt.t_interp != -1:
    results_dir += f"_t_interp_{opt.t_interp}"
    quantized_bpp = quantized_bpp/(opt.t_interp)
    temporal_interp=True
    nframes = nframes*opt.t_interp

print(f" - quantized bpp : {quantized_bpp}")
os.makedirs(results_dir, exist_ok=True)

psnrs = []
model.cuda()
model.eval()

pbar = tqdm(range(nframes))
for f in pbar:
    with torch.no_grad():
        sidelen = (resolution[0], resolution[1]) 
        total_res = resolution[0] * resolution[1]
        spatial_coord = dataio.get_mgrid(sidelen, dim=2)[None,...].cuda() 
        half_dt =  0.5 / org_nframes
        temporal_step = torch.linspace(half_dt, 1-half_dt, nframes)[f] * torch.ones(total_res)
        temporal_step = temporal_step[None,...].cuda()
        temporal_coord = torch.linspace(0, 1, nframes)[f] * torch.ones(total_res)
        temporal_coord = temporal_coord[None,...].cuda()

        all_coords = torch.cat((temporal_coord.unsqueeze(2), spatial_coord), dim=2)

        output = torch.zeros((1, total_res, 3))
        Nslice = 100
        split = int(total_res / Nslice)
        for i in range(Nslice):
            split_all_coords = all_coords[:, i*split:(i+1)*split, :]
            split_step = temporal_step[:, i*split:(i+1)*split]
            pred = model({'all_coords': split_all_coords, "temporal_steps": split_step}, temporal_interp=temporal_interp)['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()
        
        output = output.view(1, resolution[0], resolution[1], 3)
        img = output[0].permute(2, 0, 1)
        img = (img+1)/2
        img = torch.clamp(img, 0, 1)

        if opt.save:
            save_image(img, os.path.join(results_dir, f"f{str(f).zfill(5)}.png")) #[c, h, w]
 
        if opt.s_interp == -1 and opt.t_interp == -1:
            img1 = img.unsqueeze(0).cuda()

            gt = torch.Tensor(gt_frames[f, :, :, :]/255.)
            img2 = gt.unsqueeze(0).permute(0, 3, 1, 2).cuda()

            psnr = 10*torch.log10(1/torch.mean((img1-img2)**2))
            psnrs.append(psnr.item())

            pbar.set_description("Mean psnr: {:.2f}".format(sum(psnrs)/len(psnrs)))

if opt.s_interp == -1 and opt.t_interp == -1:
    mean_psnr = sum(psnrs)/len(psnrs)
    print(f"Final psnr: {mean_psnr}, bpp : {quantized_bpp}")
    file_path = os.path.join(root_path, "results", f"results.txt")
    f = open(file_path, 'a')
    f.write(f"\n - bpp : {quantized_bpp}, psnr: {mean_psnr}")
    f.write(f"\n")
    f.close()
else:
    print(f"bpp : {quantized_bpp}")
