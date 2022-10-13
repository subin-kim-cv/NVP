# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import modules
import configargparse
import torch
import numpy as np
import cv2
import math
import configargparse
import json


'''compress keyframes'''
def compress_keyframes(params, config, save_path):
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

            # quantize to 8bit (save)
            quantized_tmp_features = (tmp_features-keyframe_features_min_list[d][i])/(keyframe_features_max_list[d][i]-keyframe_features_min_list[d][i])
            quantized_tmp_features = unit_multiplier*quantized_tmp_features
            quantized_tmp_features = torch.tensor(quantized_tmp_features+0.5, dtype=torch.uint8)
            quantized_tmp_features = torch.clamp(quantized_tmp_features, 0, 255)

            resolution = resolution_in_level[i]
            quantized_tmp_features = (quantized_tmp_features.reshape(1, resolution, resolution)).permute(1, 2, 0)
            
            outputdata = np.array(quantized_tmp_features)
            cv2.imwrite(os.path.join(tmp_save_path, f"{str(i).zfill(2)}.png"), outputdata)
        
    assert check == 0

'''compress sparse grid'''
def compress_sparse_grid(params, config, save_path):
    dim = config["n_features_per_level"]
    sparse_features = params.clone().detach() # [t, h, w, d]
    t_resolution = sparse_features.shape[0]
    sparse_features_min = []
    sparse_features_max = []

    for d in range(dim):
        tmp_save_path = os.path.join(save_path, f"dim{d}")
        os.makedirs(tmp_save_path, exist_ok=True)

        tmp_sparse_features = sparse_features[:, :, :, d]
        sparse_features_min.append(torch.min(tmp_sparse_features))
        sparse_features_max.append(torch.max(tmp_sparse_features))

        # quantize to 8 bit (save)
        quantized_tmp_features = (tmp_sparse_features-sparse_features_min[d])/(sparse_features_max[d]-sparse_features_min[d])
        quantized_tmp_features = unit_multiplier*quantized_tmp_features
        quantized_tmp_features = torch.tensor(quantized_tmp_features+0.5, dtype=torch.uint8)
        quantized_tmp_features = torch.clamp(quantized_tmp_features, 0, 255)
        
        # [f, h, w, c]
        quantized_tmp_features = quantized_tmp_features.unsqueeze(3)
        outputdata = np.array(quantized_tmp_features)

        for i in range(t_resolution):
            cv2.imwrite(os.path.join(tmp_save_path, f"{str(i).zfill(5)}.png"), outputdata[i])


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


'''save trained parameters'''
keyframe_xy_saved = torch.tensor(model.keyframes_xy.params.clone().detach()).cpu()
keyframe_xt_saved = torch.tensor(model.keyframes_xt.params.clone().detach()).cpu()
keyframe_yt_saved = torch.tensor(model.keyframes_yt.params.clone().detach()).cpu()
sparse_grid_saved = torch.tensor(model.sparse_grid.embeddings.clone().detach()).cpu()


'''compress Learnable Keyframes'''
save_path_xy = os.path.join(opt.logging_root, opt.experiment_name, "compression", "src", "keyframes", "xy")
compress_keyframes(keyframe_xy_saved, config["2d_encoding_xy"], save_path_xy)

save_path_xt = os.path.join(opt.logging_root, opt.experiment_name, "compression", "src", "keyframes", "xt")
compress_keyframes(keyframe_xt_saved, config["2d_encoding_xt"], save_path_xt)

save_path_yt = os.path.join(opt.logging_root, opt.experiment_name, "compression", "src", "keyframes", "yt")
compress_keyframes(keyframe_yt_saved, config["2d_encoding_yt"], save_path_yt)

print("[i] compress learnable keyframes finished")

save_path = os.path.join(opt.logging_root, opt.experiment_name, "compression", "src", "sparsegrid")
compress_sparse_grid(sparse_grid_saved, config["3d_encoding"], save_path)

print("[i] compress sparse grid finished")