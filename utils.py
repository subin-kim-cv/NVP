import numpy as np
import torch
import dataio
import os
from torchvision.utils import make_grid

from pytorch_msssim import ms_ssim

def msssim_fn(output, target):
    if output.size(-2) >= 160:
        msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
    else:
        msssim = torch.tensor(0).to(output.device)

    return msssim

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)



def write_video_time_summary(vid_dataset, model, model_input, gt, writer, total_steps, prefix='train_'):

    resolution = vid_dataset.shape
    nframes = vid_dataset.nframes

    tmp = int(nframes/4)
    frames = [tmp*0, tmp*1, tmp*2, tmp*3]

    Nslice = 100
    with torch.no_grad():
        sidelen = (resolution[0], resolution[1])
        tot = resolution[0] * resolution[1]
        spatial_coords = [dataio.get_mgrid(sidelen, dim=2)[None,...].cuda() for f in frames]
        spatial_coords = torch.cat(spatial_coords, dim=0)

        temporal_steps = [] # modulation input
        temporal_coords = [] # temporal coords
        for idx, f in enumerate(frames):
            half_dt =  0.5 / nframes
            temporal_step = torch.linspace(half_dt, 1-half_dt, nframes)[f] * torch.ones(tot)
            temporal_coord = torch.linspace(0, 1, nframes)[f] * torch.ones(tot)
            temporal_step = temporal_step[None,...].cuda()
            temporal_steps.append(temporal_step)
            temporal_coord = temporal_coord[None,...].cuda()
            temporal_coords.append(temporal_coord)

        temporal_steps = torch.cat(temporal_steps, dim=0)
        temporal_coords = torch.cat(temporal_coords, dim=0)

        output = torch.zeros((len(frames), tot, 3))
        split = int(tot / Nslice)
        for i in range(Nslice):
            split_spatial = spatial_coords[:, i*split:(i+1)*split, :]
            split_temporal = temporal_coords[:, i*split:(i+1)*split]

            all_coords = torch.cat((split_temporal.unsqueeze(2), split_spatial), dim=2)

            split_step = temporal_steps[:, i*split:(i+1)*split]
            pred = model({'all_coords': all_coords, "temporal_steps": split_step})['model_out']
            
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[0], resolution[1], 3)
    pred_vid = pred_vid.cpu()
    pred_vid = (pred_vid+1)/2
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.Tensor(vid_dataset.vid[frames, :, :, :]/255.)
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))
    
    pred_vid = 2*pred_vid-1
    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = 2*gt_vid-1
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)

    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    min_max_summary(prefix + 'coords', model_input['all_coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_vid', pred_vid, writer, total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)

    return psnr


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)





