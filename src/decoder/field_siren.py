from torch import nn
from src.decoder.sine_layer import SineLayer

class FieldSiren(nn.Module):
    network: nn.Sequential

    def __init__(
        self,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers"""
        super().__init__()

        layers = []

        layers.append(SineLayer(d_coordinate, 256))
        layers.append(SineLayer(256, 256))
        layers.append(SineLayer(256, 256))
        layers.append(SineLayer(256, 256))
        layers.append(nn.Linear(256, d_out))

        self.model = nn.Sequential(*layers)

    def forward(self, coordinates):
        """Evaluate the MLP at the specified coordinates."""
        return self.model(coordinates)
