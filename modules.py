import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import modulation
from sparsefeaturegrid import SparseFeatureGrid
from edsr import EDSR
from decoder import MLP
       
class NVP(nn.Module):

    def __init__(self, out_features=3, encoding_config=None, latent_grid=None, **kwargs):
        super().__init__()
        n_features = encoding_config["encoder"]["n_features"]
        mlp_n_layers = encoding_config["network"]["n_hidden_layers"]
        mlp_n_neurons = encoding_config["network"]["n_neurons"]
        self.encoder = EDSR(args = encoding_config["encoder"])
        self.latent_grid = latent_grid
        self.sparse_grid = SparseFeatureGrid(level_dim=n_features, upsample=False)
        self.decoder = MLP(in_dim=n_features, out_dim=out_features, n_neurons=mlp_n_neurons, n_hidden=mlp_n_layers)


                
        # self.net = modulation.SirenNet(
        #                             dim_in = 3, # input dimension, ex. 2d coor
        #                             dim_hidden = encoding_config["network"]["n_neurons"],       # hidden dimension
        #                             dim_out = out_features,                                     # output dimension, ex. rgb value
        #                             num_layers = encoding_config["network"]["n_hidden_layers"], # number of layers
        #                             w0_initial = 30.,                                            # different signals may require 
        #                                                                                          # different omega_0 in the first layer                                                               #  - this is a hyperparameter
        #                             )
                
        # self.wrapper = modulation.SirenWrapper(self.net, latent_dim = encoding_config["encoder"]["n_features"])

    def forward(self, coords, image=None, train=True):
        if train:
            return self.forward_train(coords, image)
        else:
            return self.forward_test(coords)


    def forward_test(self, coords):
        # at test time, we do not need the image but just reconstruct ouput based in coordinates
        coords = coords.unsqueeze(0)
        net_input = self.sparse_grid.compute_features(self.latent_grid, coords)
        output = self.decoder(net_input)
        output = output.squeeze()
        return {'model_out': output}


    def forward_train(self, coords, image):
        
        self.latent_grid = self.encoder(image)

        # print('features min max: ', self.latent_grid.min(), self.latent_grid.max())

        net_input = self.sparse_grid.compute_features(self.latent_grid, coords)

        # print('net input min max: ', net_input.min(), net_input.max())

        # timesteps = model_input['temporal_steps']
        # b, t = timesteps.size(0), timesteps.size(1)
        # timesteps = timesteps.reshape(b*t, -1)

        # all_coords = model_input['all_coords']
        # all_coords = all_coords.view(-1, 3) # t, x, y

        # # keyframes
        # xy_coords = all_coords[:, [1, 2]]
        # xt_coords = all_coords[:, [0, 1]]
        # yt_coords = all_coords[:, [0, 2]]

        # spatial_embedding_xy = self.keyframes_xy(xy_coords)
        # spatial_embedding_xt = self.keyframes_xt(xt_coords)
        # spatial_embedding_yt = self.keyframes_yt(yt_coords) 

        # spatial_embedding = torch.cat((spatial_embedding_xy, spatial_embedding_yt, spatial_embedding_xt), dim=1)

        # # sparse positional features
        # if temporal_interp:
        #     motion_embedding = self.sparse_grid.forward_inter(all_coords)
        # else:
        #     motion_embedding = self.sparse_grid(all_coords)
            
        # positional features  
        # embedding = torch.cat((spatial_embedding, motion_embedding), dim=1)
        
        # modulation latent
        output = self.decoder(net_input)
        output = output.squeeze()
        # output = output.reshape((b, t, 3))

        return {'model_out': output}
    
    def set_latent_grid(self, latent_grid):
        self.latent_grid = latent_grid