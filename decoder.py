import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, n_hidden, n_neurons):
        super().__init__()
        layers = []
        lastv = in_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(lastv, n_neurons))
            layers.append(nn.ReLU())
            lastv = n_neurons
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
