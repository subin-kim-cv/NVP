import torch
from utils import make_coord
import numpy as np

class SparseFeatureGrid():
    def __init__(self, level_dim=2, upsample=False):
        self.level_dim = level_dim # latent dimension
        self.latents = None
        self.upsample = upsample
        self.local_ensemble = True

    def compute_features(self, latents, coords, decoder=None):

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            vz_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, vz_lst, eps_shift = [0], [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / latents.shape[-3] / 2
        ry = 2 / latents.shape[-2] / 2
        rz = 2 / latents.shape[-1] / 2

        # interpolate feature coordinates
        feature_coords = make_coord(latents.shape[-3:], flatten=False).cuda()
        feature_coords = feature_coords.permute(3, 0, 1, 2).unsqueeze(0)
        feature_coords = feature_coords.repeat(coords.shape[0], 1, 1, 1, 1)

        predictions = []
        volumes = []

        coords = coords.unsqueeze(1).unsqueeze(1)


        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    
                    coords_ = coords.clone()
                    coords_[..., 0] += vx * rx + eps_shift
                    coords_[..., 1] += vy * ry + eps_shift
                    coords_[..., 2] += vz * rz + eps_shift
                    coords_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    q_coords = torch.nn.functional.grid_sample(feature_coords, coords_.flip(-1), mode='nearest', align_corners=False)
                    q_coords = q_coords.squeeze(2).squeeze(2)
                    q_coords = q_coords.permute(0, 2, 1)

                    # interpolate features
                    q_features = torch.nn.functional.grid_sample(latents, coords_.flip(-1), mode='nearest', align_corners=False)
                    q_features = q_features.squeeze(2).squeeze(2)
                    q_features = q_features.permute(0, 2, 1)

                    # rel_coord = coords_.squeeze()
                    rel_coord = coords.squeeze(1).squeeze(1) - q_coords

                    # compute volume for ensemble
                    volume = torch.abs(rel_coord[..., 0] * rel_coord[..., 1] * rel_coord[..., 2])
                    volumes.append(volume + 1e-9)

                    # print("rel coord before min max: ", rel_coord.min(), rel_coord.max())
                    rel_coord[..., 0] *= latents.shape[-3]
                    rel_coord[..., 1] *= latents.shape[-2]
                    rel_coord[..., 2] *= latents.shape[-1]
        
                    input = torch.cat((q_features, rel_coord), dim=-1)
                    bs, q = coords.squeeze(1).squeeze(1).shape[:2]

                    # compute prediction for ensemble
                    prediction = decoder(input.view(bs * q, -1)).view(bs, q, -1)
                    # prediction = decoder(input)
                    predictions.append(prediction)
                    
                    # print("volume min max: ", volume.min(), volume.max())

                    

        total_volumes = torch.stack(volumes).sum(dim=0)
        
        if self.local_ensemble:
            volumes.reverse()
        
        test = 0
        out = 0
        for pred, volume in zip(predictions, volumes):
            rel_vol = (volume / total_volumes).unsqueeze(-1)
            test += volume / total_volumes
            out = out + pred * rel_vol

        return out
    




    # if self.local_ensemble:
    #         vx_lst = [-1, 1]
    #         vy_lst = [-1, 1]
    #         eps_shift = 1e-6
    #     else:
    #         vx_lst, vy_lst, eps_shift = [0], [0], 0

    #     # field radius (global: [-1, 1])
    #     rx = 2 / feat.shape[-2] / 2
    #     ry = 2 / feat.shape[-1] / 2

    #     feat_coord = make_coord(feat.shape[-2:], flatten=False) \
    #         .permute(2, 0, 1) \
    #         .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

    #     preds = []
    #     areas = []
    #     for vx in vx_lst:
    #         for vy in vy_lst:
    #             coord_ = coord.clone()
    #             coord_[:, :, 0] += vx * rx + eps_shift
    #             coord_[:, :, 1] += vy * ry + eps_shift
    #             coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

    #             q_feat = F.grid_sample(
    #                 feat, coord_.flip(-1).unsqueeze(1),
    #                 mode='nearest', align_corners=False)
                
    #             # print("qfeat shape: ", q_feat.shape)

    #             q_feat = q_feat[:, :, 0, :] \
    #                 .permute(0, 2, 1)
                

    #             q_coord = F.grid_sample(
    #                 feat_coord, coord_.flip(-1).unsqueeze(1),
    #                 mode='nearest', align_corners=False)[:, :, 0, :] \
    #                 .permute(0, 2, 1)
                


    #             rel_coord = coord - q_coord

    #             # print('coord min max: ', coord.min(), coord.max())
    #             # print('q_coord min max: ', q_coord.min(), q_coord.max())
    #             # print("rel coords min max before: ", rel_coord.min(), rel_coord.max())

    #             rel_coord[:, :, 0] *= feat.shape[-2]
    #             rel_coord[:, :, 1] *= feat.shape[-1]

    #             # print("-------------------------------")

    #             # print("Cord min max: ", coord.min(), coord.max())
    #             # print("Qcord min max: ", q_coord.min(), q_coord.max())
    #             # print("Real coord min max: ", rel_coord.min(), rel_coord.max())
    #             # print("Qfeat min max: ", q_feat.min(), q_feat.max())


    #             inp = torch.cat([q_feat, rel_coord], dim=-1)

    #             if self.cell_decode:
    #                 rel_cell = cell.clone()
    #                 rel_cell[:, :, 0] *= feat.shape[-2]
    #                 rel_cell[:, :, 1] *= feat.shape[-1]
    #                 inp = torch.cat([inp, rel_cell], dim=-1)

    #             bs, q = coord.shape[:2]

    #             # np_inp = inp.cpu().detach().numpy()
        
    #             # # Compute the histogram
    #             # hist, bin_edges = np.histogram(np_inp, bins=20)

    #             # print("Net input histogram ....")

    #             # # Print the histogram
    #             # for b in range(len(bin_edges)-1):
    #             #     print(f"Bin: {bin_edges[b]:.2f} to {bin_edges[b+1]:.2f}, Count: {hist[b]}")

    #             # print("----------------------------------------------------")


    #             pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
    #             preds.append(pred)

    #             area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
    #             areas.append(area + 1e-9)

    #     tot_area = torch.stack(areas).sum(dim=0)
    #     if self.local_ensemble:
    #         t = areas[0]; areas[0] = areas[3]; areas[3] = t
    #         t = areas[1]; areas[1] = areas[2]; areas[2] = t
    #     ret = 0
    #     for pred, area in zip(preds, areas):
    #         ret = ret + pred * (area / tot_area).unsqueeze(-1)
    #     return ret