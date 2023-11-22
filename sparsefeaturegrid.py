import torch
from utils import make_coord

class SparseFeatureGrid():
    def __init__(self, level_dim=2, upsample=False):
        self.level_dim = level_dim # latent dimension
        self.latents = None
        self.upsample = upsample

    def compute_features(self, latents, coords):
        coords_ = coords.clone()
        coords = coords.clamp_(-1 + 1e-6, 1 - 1e-6)
        coords = coords.unsqueeze(1).unsqueeze(1)

        # interpolate feature coordinates
        feature_coords = make_coord(latents.shape[-3:], flatten=False).cuda()
        feature_coords = feature_coords.permute(3, 0, 1, 2).unsqueeze(0)
        feature_coords = feature_coords.repeat(coords.shape[0], 1, 1, 1, 1)    
        q_coords = torch.nn.functional.grid_sample(feature_coords, coords, mode='bilinear', align_corners=False)
        q_coords = q_coords.squeeze()
        q_coords = q_coords.permute(0, 2, 1)

        # interpolate features
        q_features = torch.nn.functional.grid_sample(latents, coords, mode='bilinear', align_corners=False)
        q_features = q_features.squeeze()
        q_features = q_features.permute(0, 2, 1)
        
        out = torch.cat((q_features, q_coords), dim=-1)
        return out