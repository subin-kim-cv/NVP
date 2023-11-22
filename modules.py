from torch import nn
import torch.nn.functional as F
from sparsefeaturegrid import SparseFeatureGrid
from edsr import EDSR
from decoder import MLP
class NVP(nn.Module):

    def __init__(self, out_features=3, encoding_config=None, latent_grid=None, **kwargs):
        super().__init__()

        self.feat_unfold = True

        # hyper parameters
        n_features = encoding_config["encoder"]["n_features"]
        mlp_n_layers = encoding_config["network"]["n_hidden_layers"]
        mlp_n_neurons = encoding_config["network"]["n_neurons"]

        model_in = n_features + 3 

        # module for latent grid processing
        self.latent_grid = latent_grid
        self.sparse_grid = SparseFeatureGrid(level_dim=n_features, upsample=False)

        # trainable parameters
        self.encoder = EDSR(args = encoding_config["encoder"])
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=mlp_n_neurons, n_hidden=mlp_n_layers)

    def forward(self, coords, image=None, train=True):
        if train:
            return self.forward_train(coords, image)
        else:
            return self.forward_test(coords)


    def forward_test(self, coords):
        # at test time, we do not need the image but just reconstruct ouput based in coordinates
        net_input = self.sparse_grid.compute_features(self.latent_grid, coords)
        output = self.decoder(net_input)
        output = output.squeeze()
        return {'model_out': output}


    def forward_train(self, coords, image):        
        self.latent_grid = self.encoder(image)
        net_input = self.sparse_grid.compute_features(self.latent_grid, coords)
        output = self.decoder(net_input)
        output = output.squeeze()
        return {'model_out': output}
    
    def set_latent_grid(self, latent_grid):
        self.latent_grid = latent_grid