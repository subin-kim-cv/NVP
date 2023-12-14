import torch
from utils import make_coord

class SparseFeatureGrid2D():
    def __init__(self, feat_unfold, local_ensemble, upsample=False):
        self.latents = None
        self.upsample = upsample
        self.local_ensemble = local_ensemble
        self.feature_unfold = feat_unfold
    
    def compute_features(self, latents, coords, decoder):
        if self.feature_unfold:
            # concat each latent by it's local neighborhood
            latents = self.unfold_features(latents)

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / latents.shape[-2] / 2
        ry = 2 / latents.shape[-1] / 2

        # interpolate feature coordinates
        feature_coords = make_coord(latents.shape[-2:], flatten=False).cuda()
        feature_coords = feature_coords.permute(2, 0, 1).unsqueeze(0)
        feature_coords = feature_coords.repeat(coords.shape[0], 1, 1, 1)

        predictions = []
        volumes = []

        coords = coords.unsqueeze(1)


        for vx in vx_lst:
            for vy in vy_lst:
                    
                    coords_ = coords.clone()
                    coords_[..., 0] += vx * rx + eps_shift
                    coords_[..., 1] += vy * ry + eps_shift
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
                    volume = torch.abs(rel_coord[..., 0] * rel_coord[..., 1])
                    volumes.append(volume + 1e-9)

                    rel_coord[..., 0] *= latents.shape[-2]
                    rel_coord[..., 1] *= latents.shape[-1]

                    input = torch.cat((q_features, rel_coord), dim=-1)
                    bs, q = coords.squeeze(1).squeeze(1).shape[:2]

                    # compute prediction for ensemble
                    prediction = decoder(input.view(bs * q, -1)).view(bs, q, -1)
                    predictions.append(prediction)
                    

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

class SparseFeatureGrid():
    def __init__(self, feat_unfold, local_ensemble, upsample=False):
        self.latents = None
        self.upsample = upsample
        self.local_ensemble = local_ensemble
        self.feature_unfold = feat_unfold

    def unfold_features(self, latents):
        unfold_list = [-1, 0, 1]
        bs, n_feat, x_dim, y_dim, z_dim = latents.shape

        x_idx = torch.arange(0, latents.shape[-3], dtype=torch.int64).cuda()
        y_idx = torch.arange(0, latents.shape[-2], dtype=torch.int64).cuda()
        z_idx = torch.arange(0, latents.shape[-1], dtype=torch.int64).cuda()

        tmp_x, tmp_y, tmp_z = torch.meshgrid(x_idx, y_idx, z_idx)

        idx = torch.stack((tmp_x, tmp_y, tmp_z), dim=-1)

        x = idx[..., 0].flatten()
        y = idx[..., 1].flatten()
        z = idx[..., 2].flatten()

        neighbors = []

        for i in unfold_list:
            for j in unfold_list:
                for k in unfold_list:
                    vx = torch.clamp(x+i, 0, latents.shape[-3]-1)
                    vy = torch.clamp(y+j, 0, latents.shape[-2]-1)
                    vz = torch.clamp(z+k, 0, latents.shape[-1]-1)

                    nbh = latents[:, :, vx, vy, vz]
                    neighbors.append(nbh)

        latents = torch.stack(neighbors, dim=1).view(bs, n_feat * len(unfold_list)**3, x_dim, y_dim, z_dim)
        return latents

    def compute_features(self, latents, coords, decoder=None):

        if self.feature_unfold:
            # concat each latent by it's local neighborhood
            latents = self.unfold_features(latents)

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

                    rel_coord[..., 0] *= latents.shape[-3]
                    rel_coord[..., 1] *= latents.shape[-2]
                    rel_coord[..., 2] *= latents.shape[-1]

                    zeros = torch.zeros(rel_coord.size(0), rel_coord.size(1), 1).cuda()

                    rel_coord = torch.cat((rel_coord, zeros), dim=-1)

                    input = torch.cat((q_features, rel_coord), dim=-1)
                    bs, q = coords.squeeze(1).squeeze(1).shape[:2]

                    # compute prediction for ensemble
                    prediction = decoder(input.view(bs * q, -1)).view(bs, q, -1)
                    predictions.append(prediction)
                    

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