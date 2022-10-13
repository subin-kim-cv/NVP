import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import modulation
from sparsegrid import SparseGrid
       
class NVP(nn.Module):

    def __init__(self, out_features=3, encoding_config=None, **kwargs):
        super().__init__()

        # learnable keyframes xy
        self.keyframes_xy = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config["2d_encoding_xy"])    
        assert self.keyframes_xy.dtype == torch.float32

        # learnable keyframes yt
        self.keyframes_yt = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config["2d_encoding_yt"]) # torch.Tensor       
        assert self.keyframes_yt.dtype == torch.float32

        # learnable keyframes xt
        self.keyframes_xt = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config["2d_encoding_xt"]) # torch.Tensor     
        assert self.keyframes_xt.dtype == torch.float32

        self.sparse_grid = SparseGrid(level_dim=encoding_config["3d_encoding"]["n_features_per_level"], 
                                    x_resolution=encoding_config["3d_encoding"]["x_resolution"],
                                    y_resolution=encoding_config["3d_encoding"]["y_resolution"],
                                    t_resolution=encoding_config["3d_encoding"]["t_resolution"], 
                                    upsample=encoding_config["3d_encoding"]["upsample"]
                                    )
                
        self.net = modulation.SirenNet(
                                    dim_in = 1, # input dimension, ex. 2d coor
                                    dim_hidden = encoding_config["network"]["n_neurons"],       # hidden dimension
                                    dim_out = out_features,                                     # output dimension, ex. rgb value
                                    num_layers = encoding_config["network"]["n_hidden_layers"], # number of layers
                                    w0_initial = 30.,                                            # different signals may require 
                                                                                                 # different omega_0 in the first layer                                                               #  - this is a hyperparameter
                                    )
                
        # keyframes + sparse grid
        latent_dim = encoding_config["2d_encoding_xy"]["n_levels"]*(encoding_config["2d_encoding_xy"]["n_features_per_level"])
        latent_dim += encoding_config["2d_encoding_yt"]["n_levels"]*(encoding_config["2d_encoding_yt"]["n_features_per_level"])
        latent_dim += encoding_config["2d_encoding_xt"]["n_levels"]*(encoding_config["2d_encoding_xt"]["n_features_per_level"])
        latent_dim += (encoding_config["3d_encoding"]["n_features_per_level"])*9

        self.wrapper = modulation.SirenWrapper(self.net, latent_dim = latent_dim)

        print(self)

    def forward(self, model_input, temporal_interp=False, params=None):

        timesteps = model_input['temporal_steps']
        b, t = timesteps.size(0), timesteps.size(1)
        timesteps = timesteps.reshape(b*t, -1)

        all_coords = model_input['all_coords']
        all_coords = all_coords.view(-1, 3) # t, x, y

        # keyframes
        xy_coords = all_coords[:, [1, 2]]
        xt_coords = all_coords[:, [0, 1]]
        yt_coords = all_coords[:, [0, 2]]

        spatial_embedding_xy = self.keyframes_xy(xy_coords)
        spatial_embedding_xt = self.keyframes_xt(xt_coords)
        spatial_embedding_yt = self.keyframes_yt(yt_coords) 

        spatial_embedding = torch.cat((spatial_embedding_xy, spatial_embedding_yt, spatial_embedding_xt), dim=1)

        # sparse positional features
        if temporal_interp:
            motion_embedding = self.sparse_grid.forward_inter(all_coords)
        else:
            motion_embedding = self.sparse_grid(all_coords)
            
        # positional features  
        embedding = torch.cat((spatial_embedding, motion_embedding), dim=1)
        
        # modulation latent
        output = self.wrapper(coords=timesteps, latent=embedding)
        output = output.reshape((b, t, 3))

        return {'model_out': output}