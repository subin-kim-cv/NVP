import torch
import torch.nn as nn

class SparseFeatureGrid():
    def __init__(self, level_dim=2, upsample=False):
        self.level_dim = level_dim # latent dimension
        self.latents = None
        self.upsample = upsample

    def compute_features(self, latents, inputs):
        # inputs [0, 1]
        if self.upsample:
            # upsampling sparse positional features
            tmp_embeddings = latents.permute(3, 0, 1, 2) # dim, T, H, W
            tmp_embeddings = tmp_embeddings.unsqueeze(0)
            tmp_embeddings = torch.nn.functional.interpolate(tmp_embeddings, scale_factor=2, mode='trilinear')
            tmp_embeddings = tmp_embeddings.squeeze()
            tmp_embeddings = tmp_embeddings.permute(1, 2, 3, 0)
        else:
            tmp_embeddings = latents

        # tmp_embeddings = tmp_embeddings.permute(0, 2, 3, 4, 1)
        inputs = inputs.unsqueeze(1).unsqueeze(1)

        tmp_embeddings = 2.0 * ((tmp_embeddings - tmp_embeddings.min()) / (tmp_embeddings.max() - tmp_embeddings.min())) - 1
        # print("min and max of grid", tmp_embeddings.min(), tmp_embeddings.max())
        grid_features = torch.nn.functional.grid_sample(tmp_embeddings, inputs, mode='bilinear', padding_mode='reflection')
        grid_features = grid_features.squeeze()
        grid_features = grid_features.permute(0, 2, 1)

        # round down
        # t_coord = inputs[:, 0]
        # t_coord_float = ((t_res-1)*t_coord)
        # t_coord_idx = (t_coord_float+0.5).type(torch.int64)
        # t_coord_idx = torch.clamp(t_coord_idx, 0, t_res-1)

        # x_coord = inputs[:, 1]
        # x_coord_float = ((x_res-1)*x_coord)
        # x_coord_idx = (x_coord_float+0.5).type(torch.int64)
        # x_coord_idx = torch.clamp(x_coord_idx, 0, x_res-1)

        # y_coord = inputs[:, 2]
        # y_coord_float = ((y_res-1)*y_coord)
        # y_coord_idx = (y_coord_float+0.5).type(torch.int64)
        # y_coord_idx = torch.clamp(y_coord_idx, 0, y_res-1)

        # grid_features = None
        # unfold_list = [-1, 0, 1]

        # for i in unfold_list:
        #     for j in unfold_list:
        #         vx = torch.clamp(x_coord_idx+i, 0, x_res-1)
        #         vy = torch.clamp(y_coord_idx+j, 0, y_res-1)
        #         feat = (tmp_embeddings[t_coord_idx, vx, vy, :])
        #         if grid_features == None:
        #             grid_features = feat
        #         else:
        #             grid_features = torch.cat((grid_features, feat), dim=1)


        return grid_features