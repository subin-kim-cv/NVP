# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, modules
import configargparse
import skvideo.datasets
import torch
from tqdm import tqdm
from torch import nn
from PIL import Image
import numpy as np
import math

import configargparse
import skvideo.datasets
import json
from torchvision.utils import save_image


def load_compressed_keyframes(params, config, save_path, qscale):
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

    tmp_all = []
    tmp = None
    num_bits = 0
    for d in range(dim):
        tmp_save_path = os.path.join(save_path, f"dim{d}")
        os.makedirs(tmp_save_path, exist_ok=True)

        keyframe_features_max_list.append([])
        keyframe_features_min_list.append([])
        for i in range(n_levels):
            if i != n_levels-1:
                pos1 = params_in_level[i]
                pos2 = params_in_level[i+1]
                tmp_features = keyframe_features_list[d][pos1:pos2]
            else:
                pos1 = params_in_level[i]
                tmp_features = keyframe_features_list[d][pos1:]
            
            check -= len(tmp_features)
            
            keyframe_features_max_list[d].append(torch.max(tmp_features))
            keyframe_features_min_list[d].append(torch.min(tmp_features))
    

        load_path = os.path.join(tmp_save_path, str(qscale).zfill(3))
        for i in range(n_levels):
            img_path = load_path+"/"+str(i).zfill(2)+".jpg"
            num_bits += os.path.getsize(img_path)*8
            img = Image.open(img_path).convert("L")
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32).reshape(-1)
            img = img/unit_multiplier
            img = (keyframe_features_max_list[d][i]-keyframe_features_min_list[d][i])*img+keyframe_features_min_list[d][i]

            if i == 0:
                tmp = img
            else:
                tmp = torch.cat((tmp, img), dim=-1)
                    
        tmp_all.append(tmp)
            
    tmp_content = torch.stack(tmp_all, dim=-1)
    tmp_content = tmp_content.reshape(-1)

    return nn.Parameter(tmp_content.cuda()), num_bits

def load_compressed_sparse_grid(params, config, save_path, crf, framerate):
    dim = config["n_features_per_level"]
    sparse_features = params.clone().detach() # [t, h, w, d]
    t_resolution = sparse_features.shape[0]
    sparse_features_min = []
    sparse_features_max = []

    tmp_mp4_all = None
    num_bits = 0
    for d in range(dim):
        tmp_save_path = os.path.join(save_path, f"dim{d}", f"{crf}_{framerate}.mp4")

        tmp_sparse_features = sparse_features[:, :, :, d]
        sparse_features_min.append(torch.min(tmp_sparse_features))
        sparse_features_max.append(torch.max(tmp_sparse_features))
        
        num_bits += os.path.getsize(tmp_save_path)*8
        tmp_mp4 = skvideo.io.vread(tmp_save_path, outputdict={"-pix_fmt": "gray"}) 

        
        # to 32 bit
        tmp_mp4 = tmp_mp4/unit_multiplier
        tmp_mp4 = (sparse_features_max[d]-sparse_features_min[d])*tmp_mp4+sparse_features_min[d]
        if tmp_mp4_all == None:
            tmp_mp4_all = tmp_mp4
        else:
            tmp_mp4_all = torch.cat((tmp_mp4_all, tmp_mp4), dim=3)
        
    return nn.Parameter(torch.tensor(tmp_mp4_all, dtype=torch.float32).cuda()), num_bits

#############################################################################################################

command_line = ""
for arg in sys.argv:
    command_line += arg
    command_line += " "

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=True,  help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs_NVP', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False, default="",
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

'''Compression configure'''
p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g., /data/UVG/Jockey)")
p.add_argument('--num_frames', type=int, default=600, required=True, help="Number of video frames to reconstruct")
p.add_argument("--qscale", nargs='+', required=True, help="Compression qscale list of (U_xy, U_xt, U_yt) (e.g., 2 3 3)")
p.add_argument("--crf", type=int, required=True, help="Compression crf (e.g., 21)")
p.add_argument("--framerate", type=int, required=True, help="Compression framerate (e.g., 25)")

'''Save configure'''
p.add_argument("--save", default=False, action="store_true", help="Save the Compressed frames")
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
model.load_state_dict(checkpoint['model'])


'''Save trained parameters'''
keyframe_xy_saved = torch.tensor(model.keyframes_xy.params.clone().detach()).cpu()
keyframe_xt_saved = torch.tensor(model.keyframes_xt.params.clone().detach()).cpu()
keyframe_yt_saved = torch.tensor(model.keyframes_yt.params.clone().detach()).cpu()
sparse_grid_saved = torch.tensor(model.sparse_grid.embeddings.clone().detach()).cpu()


'''Load Compressed Learnable Keyframes'''
save_path_xy = os.path.join(opt.logging_root, opt.experiment_name, "compression", "dst", "keyframes", "xy")
model.keyframes_xy.params, xy_bits = load_compressed_keyframes(keyframe_xy_saved, config["2d_encoding_xy"], save_path_xy, opt.qscale[0])

save_path_xt = os.path.join(opt.logging_root, opt.experiment_name, "compression", "dst", "keyframes", "xt")
model.keyframes_xt.params, xt_bits = load_compressed_keyframes(keyframe_xt_saved, config["2d_encoding_xt"], save_path_xt, opt.qscale[1])

save_path_yt = os.path.join(opt.logging_root, opt.experiment_name, "compression", "dst", "keyframes", "yt")
model.keyframes_yt.params, yt_bits = load_compressed_keyframes(keyframe_yt_saved, config["2d_encoding_yt"], save_path_yt, opt.qscale[2])

print("[i] Loading learnable keyframes finished")


'''Load Compressed Sparse Grid'''
save_path = os.path.join(opt.logging_root, opt.experiment_name, "compression", "dst", "sparsegrid")
model.sparse_grid.embeddings, sparsegrid_bits = load_compressed_sparse_grid(sparse_grid_saved, config["3d_encoding"], save_path, opt.crf, opt.framerate)

print("[i] Loading sparse grid finished")


'''Load Video Dataset'''
video_path = opt.dataset
vid_dataset = dataio.VideoTime(video_path, split_num=opt.num_frames)

''' BPP Calculation '''
compressed_bits = xy_bits+xt_bits+yt_bits+sparsegrid_bits
mlp_bits = (utils.get_n_params(model.wrapper))*32
tot_bits = compressed_bits+mlp_bits
new_bpp = tot_bits/(vid_dataset.vid.shape[0]*vid_dataset.vid.shape[1]*vid_dataset.vid.shape[2])

print(f" - JPEG qscale: {opt.qscale}")
print(f" - HEVC framerate: {opt.framerate}")
print(f" - HEVC crf: {opt.crf}")
print(f" - Bpp : {new_bpp}")


results = {}
results_dir = os.path.join(root_path, 'results')
os.makedirs(results_dir, exist_ok=True)

# Get ground truth and input data
gt_frames = vid_dataset.vid
resolution = vid_dataset.shape
nframes = vid_dataset.nframes

psnrs = []
model.cuda()
model.eval()

pbar = tqdm(range(nframes))
for f in pbar:
    with torch.no_grad():
        sidelen = (resolution[0], resolution[1]) 
        total_res = resolution[0] * resolution[1]
        spatial_coord = dataio.get_mgrid(sidelen, dim=2)[None,...].cuda() 
        half_dt =  0.5 / nframes
        temporal_step = torch.linspace(half_dt, 1-half_dt, nframes)[f] * torch.ones(total_res)
        temporal_step = temporal_step[None,...].cuda()
        temporal_coord = torch.linspace(0, 1, nframes)[f] * torch.ones(total_res)
        temporal_coord = temporal_coord[None,...].cuda()

        all_coords = torch.cat((temporal_coord.unsqueeze(2), spatial_coord), dim=2)
                    
        pred = model({'all_coords': all_coords, "temporal_steps": temporal_step})['model_out']

        pred_saved = pred.cpu() # [-1, 1]

        # save image
        if opt.save:
            img = pred[0].reshape(resolution[0], resolution[1], 3).permute(2, 0, 1)
            img = (img+1)/2
            img = torch.clamp(img, 0, 1)
            save_image(img, os.path.join(results_dir, f"f{str(f).zfill(5)}.png")) #[c, h, w]
                    

        pred = pred[0].reshape(resolution[0], resolution[1], 3).permute(2, 0, 1)
        pred = (pred+1)/2
        pred = torch.clamp(pred, 0, 1)
        img1 = pred.unsqueeze(0).cuda()

        gt = torch.Tensor(gt_frames[f, :, :, :]/255.)
        img2 = gt.unsqueeze(0).permute(0, 3, 1, 2).cuda()

        psnr = 10*torch.log10(1/torch.mean((img1-img2)**2))
        psnrs.append(psnr.item())
        
        pbar.set_description("Mean psnr: {:.2f}".format(sum(psnrs)/len(psnrs)))

mean_psnr = sum(psnrs)/len(psnrs)
print(f"Final psnr: {mean_psnr}, bpp : {new_bpp}")

file_path = os.path.join(root_path, "results", f"results.txt")
f = open(file_path, 'a')
f.write(f"\n - bpp : {new_bpp}, psnr: {mean_psnr}")
f.write(f"\n")
f.close()
