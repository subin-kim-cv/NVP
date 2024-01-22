import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        is_first: bool = True,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.d_in = d_in
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.d_in, 1 / self.d_in)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.d_in) / self.omega_0,
                    np.sqrt(6 / self.d_in) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))