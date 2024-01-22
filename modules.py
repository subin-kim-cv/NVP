from torch import nn
from sparsefeaturegrid import SparseFeatureGrid, SparseFeatureGrid2D
from edsr_2d import EDSR2D
from edsr import EDSR
from decoder import MLP
from src.encoders import rdn
from src.encoders import swinir
from src.decoder import inr
from src.decoder.field_siren import FieldSiren
import os

class CVR(nn.Module):
    def __init__(self, out_features=3, encoding_config=None, latent_grid=None, export_features=False, **kwargs):
        super().__init__()

        self.feat_unfold = False
        self.local_ensemble = False
        self.pos_enc = False
        self.export_features = export_features

        # hyper parameters
        n_features = encoding_config["encoder"]["n_features"]
        mlp_n_layers = encoding_config["network"]["n_hidden_layers"]
        mlp_n_neurons = encoding_config["network"]["n_neurons"]


        model_in = n_features + 2 
        if self.feat_unfold:
            model_in = n_features * (3**2) + 2 # expand features by local neighborhood

        # module for latent grid processing
        self.latent_grid = latent_grid
        self.sparse_grid = SparseFeatureGrid2D(feat_unfold=self.feat_unfold, local_ensemble=self.local_ensemble, upsample=False)

        # trainable parameters
        self.encoder = EDSR2D(args = encoding_config["encoder"])
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        # self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=mlp_n_neurons, n_hidden=mlp_n_layers, pos_enc=self.pos_enc)
        # self.decoder = inr.Gabor(in_features=model_in, hidden_features=mlp_n_neurons, hidden_layers=mlp_n_layers, out_features=out_features)
        self.decoder = FieldSiren(d_coordinate=model_in, d_out=out_features)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)
        output = self.sparse_grid.compute_features(latent_grid, coords, self.decoder)
        return {'model_out': output}

class NVP(nn.Module):

    def __init__(self, out_features=3, encoding_config=None, latent_grid=None, export_features=False, **kwargs):
        super().__init__()

        self.feat_unfold = False
        self.local_ensemble = False
        self.export_features = export_features

        # hyper parameters
        n_features = encoding_config["encoder"]["n_features"]
        mlp_n_layers = encoding_config["network"]["n_hidden_layers"]
        mlp_n_neurons = encoding_config["network"]["n_neurons"]


        model_in = n_features + 4 
        if self.feat_unfold:
            model_in = n_features * (3**3) + 3 # expand features by local neighborhood

        # module for latent grid processing
        self.latent_grid = latent_grid
        self.sparse_grid = SparseFeatureGrid(feat_unfold=self.feat_unfold, local_ensemble=self.local_ensemble, upsample=False)

        # trainable parameters
        self.encoder = EDSR(args = encoding_config["encoder"])
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=mlp_n_neurons, n_hidden=mlp_n_layers)

    def forward(self, coords, image=None, train=True, chunk_position=None):
        # if train:
        return self.forward_train(coords, image, chunk_position)
        # else:
        #     return self.forward_test(coords)


    # def forward_test(self, coords):
    #     # at test time, we do not need the image but just reconstruct ouput based in coordinates
    #     output = self.sparse_grid.compute_features(self.latent_grid, coords, self.decoder)
    #     # output = self.decoder(net_input)
    #     output = output.squeeze()
    #     return {'model_out': output}


    def forward_train(self, coords, image, chunk_position=None):        
        self.latent_grid = self.encoder(image)

        # if self.export_features:
        #     print("Features shape: ", self.latent_grid.shape)

        #     latent_to_export = self.latent_grid.squeeze().permute(1, 2, 3, 0)

        #     # flatten and export to JSON
        #     exp_features = latent_to_export.cpu().detach().numpy()
        #     exp_features = exp_features.flatten().tolist()

        #     output = {}
        #     output["shape"] = latent_to_export.shape
        #     output["position"] = chunk_position.tolist()
        #     output["data"] = exp_features

        #     with open("{}_{}_{}_features.json".format(chunk_position[0], chunk_position[1], chunk_position[2]), 'w') as outfile:
        #         json.dump(output, outfile)



        # print('Features min max: ', self.latent_grid.min(), self.latent_grid.max())
        output = self.sparse_grid.compute_features(self.latent_grid, coords, self.decoder)
        output = output.squeeze(-1)
        return {'model_out': output}
    
    def set_latent_grid(self, latent_grid):
        self.latent_grid = latent_grid

    def export_onnx(self, path):

        # todo export feature grid
        # here

        # export decoder
        mlp_name = os.path.join(path, 'mlp.json')
        self.decoder.export(mlp_name)

