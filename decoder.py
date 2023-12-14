import torch.nn as nn
import torch
import json

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, n_hidden, n_neurons):
        super().__init__()
        layers = []
        self.in_dim = in_dim
        self.out_dim = out_dim
        lastv = in_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(lastv, n_neurons))
            layers.append(nn.ReLU())
            lastv = n_neurons
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    def export(self, filename):
        # export model
        self.eval()

        dummy_input = torch.ones(1, self.in_dim).cuda()
        dummy_out = self(dummy_input)


        activation = "Relu"
        weights_and_biases = {}
        weights_and_biases['input_shape'] = [None, self.in_dim]
        weights_and_biases['output_shape'] = [None, self.out_dim]
        weights_and_biases['activations'] = activation
        weights_and_biases['dummy_input'] = dummy_input.cpu().detach().numpy().tolist()
        weights_and_biases['dummy_output'] = dummy_out.cpu().detach().numpy().tolist()

        layers = {}
        for name, param in self.named_parameters():

            name_parts = name.split('.')
            key = name_parts[0] + "." + name_parts[1]
            if key not in layers:
                layers[key] = {}
            
            param_np = param.cpu().detach().numpy()
            layers[key][name_parts[2]] = param_np.flatten(order="F").tolist()
            layers[key][name_parts[2] + '_shape'] = list(param_np.shape)

        sorted_keys = sorted(layers.keys())
        weights_and_biases['layers'] = [layers[key] for key in sorted_keys]

        # safe weights and biases as json
        with open(filename, 'w') as outfile:
            json.dump(weights_and_biases, outfile)



    def export_onnx(self, filename):
        # export onnx model
        self.eval()
        input = torch.randn(1, self.in_dim).cuda()
        torch.onnx.export(self.layers, input, filename)
