import torch
import torch.nn as nn

class SparseGrid(nn.Module):
    def __init__(self, level_dim=2, x_resolution=300, y_resolution=300, t_resolution=600, upsample=False):
        super().__init__()

        self.level_dim = level_dim # latent dimension

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.t_resolution = t_resolution
        self.embeddings = nn.Parameter(torch.empty(self.t_resolution, self.x_resolution, self.y_resolution, self.level_dim))

        self.upsample = upsample

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)
    
    def forward(self, inputs):

        # inputs [0, 1]
        if self.upsample:
            # upsampling sparse positional features
            tmp_embeddings = self.embeddings.permute(3, 0, 1, 2) # dim, T, H, W
            tmp_embeddings = torch.nn.functional.interpolate(tmp_embeddings, scale_factor=2, mode='bilinear')
            tmp_embeddings = tmp_embeddings.permute(1, 2, 3, 0)
            tmp_shape = tmp_embeddings.shape
            t_res = self.t_resolution
            x_res = tmp_shape[1]
            y_res = tmp_shape[2]

        else:
            t_res = self.t_resolution
            x_res = self.x_resolution
            y_res = self.y_resolution
            tmp_embeddings = self.embeddings

        # round down
        t_coord = inputs[:, 0]
        t_coord_float = ((t_res-1)*t_coord)
        t_coord_idx = (t_coord_float+0.5).type(torch.int64)
        t_coord_idx = torch.clamp(t_coord_idx, 0, t_res-1)

        x_coord = inputs[:, 1]
        x_coord_float = ((x_res-1)*x_coord)
        x_coord_idx = (x_coord_float+0.5).type(torch.int64)
        x_coord_idx = torch.clamp(x_coord_idx, 0, x_res-1)

        y_coord = inputs[:, 2]
        y_coord_float = ((y_res-1)*y_coord)
        y_coord_idx = (y_coord_float+0.5).type(torch.int64)
        y_coord_idx = torch.clamp(y_coord_idx, 0, y_res-1)

        grid_features = None
        unfold_list = [-1, 0, 1]

        for i in unfold_list:
            for j in unfold_list:
                vx = torch.clamp(x_coord_idx+i, 0, x_res-1)
                vy = torch.clamp(y_coord_idx+j, 0, y_res-1)
                feat = (tmp_embeddings[t_coord_idx, vx, vy, :])
                if grid_features == None:
                    grid_features = feat
                else:
                    grid_features = torch.cat((grid_features, feat), dim=1)


        return grid_features



    def forward_inter(self, inputs):

        # inputs [0, 1]
        if self.upsample:
            # upsampling sparse positional features
            tmp_embeddings = self.embeddings.permute(3, 0, 1, 2) # dim, T, H, W
            tmp_embeddings = torch.nn.functional.interpolate(tmp_embeddings, scale_factor=2, mode='bilinear')
            tmp_embeddings = tmp_embeddings.permute(1, 2, 3, 0)
            tmp_shape = tmp_embeddings.shape
            t_res = self.t_resolution
            x_res = tmp_shape[1]
            y_res = tmp_shape[2]

        else:
            t_res = self.t_resolution
            x_res = self.x_resolution
            y_res = self.y_resolution
            tmp_embeddings = self.embeddings


        # round down
        t_coord = inputs[:, 0]
        t_coord_float = ((t_res-1)*t_coord)
        t_coord_idx = (t_coord_float+0.5).type(torch.int64)
        t_coord_idx = torch.clamp(t_coord_idx, 0, t_res-1)

        t_coord_idx_lower = t_coord_float.type(torch.int64)
        t_coord_idx_upper = (t_coord_float+1).type(torch.int64)
        t_coord_idx_upper = torch.clamp(t_coord_idx_upper, 0, t_res-1)
        upper_coeff = t_coord_float-t_coord_idx_lower
        lower_coeff = t_coord_idx_upper-t_coord_float

        upper_coeff = upper_coeff/(upper_coeff+lower_coeff)
        lower_coeff = lower_coeff/(upper_coeff+lower_coeff)


        x_coord = inputs[:, 1]
        x_coord_float = ((x_res-1)*x_coord)
        x_coord_idx = (x_coord_float+0.5).type(torch.int64)
        x_coord_idx = torch.clamp(x_coord_idx, 0, x_res-1)

        y_coord = inputs[:, 2]
        y_coord_float = ((y_res-1)*y_coord)
        y_coord_idx = (y_coord_float+0.5).type(torch.int64)
        y_coord_idx = torch.clamp(y_coord_idx, 0, y_res-1)

        unfold_list = [-1, 0, 1]

        grid_features_1 = None
        lower_coeff = lower_coeff.unsqueeze(1)
        lower_coeff = lower_coeff.repeat(1, self.level_dim)

        for i in unfold_list:
            for j in unfold_list:
                vx = torch.clamp(x_coord_idx+i, 0, x_res-1)
                vy = torch.clamp(y_coord_idx+j, 0, y_res-1)
                feat = (self.embeddings[t_coord_idx_lower, vx, vy, :])
                feat *= lower_coeff
                if grid_features_1 == None:
                    grid_features_1 = feat
                else:
                    grid_features_1 = torch.cat((grid_features_1, feat), dim=1)


        grid_features_2 = None
        upper_coeff = upper_coeff.unsqueeze(1)
        upper_coeff = upper_coeff.repeat(1, self.level_dim)
        for i in unfold_list:
            for j in unfold_list:
                vx = torch.clamp(x_coord_idx+i, 0, x_res-1)
                vy = torch.clamp(y_coord_idx+j, 0, y_res-1)
                feat = (self.embeddings[t_coord_idx_upper, vx, vy, :])
                feat *= upper_coeff
                if grid_features_2 == None:
                    grid_features_2 = feat
                else:
                    grid_features_2 = torch.cat((grid_features_2, feat), dim=1)

        grid_features = grid_features_1+grid_features_2

        return grid_features



