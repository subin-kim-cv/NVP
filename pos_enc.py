import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves

    def forward(self, samples):
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)

        freq = (
            2 ** torch.arange(1, self.num_octaves + 1, dtype=torch.float32) * torch.pi
        ).to(samples.device)

        freq = freq.view(1, 1, -1)  # prepare for broadcasting
        samples = samples[..., None]  # prepare for broadcasting

        phase = samples * freq

        phase_shape = list(phase.shape)
        phase_transformed_shape = phase_shape[:-2] + [phase_shape[-2] * phase_shape[-1]]
        sin = torch.sin(phase).reshape(*phase_transformed_shape)
        cos = torch.cos(phase).reshape(*phase_transformed_shape)

        output = torch.stack([sin, cos], dim=-1)

        shape = list(output.shape)
        transformed_shape = shape[:-2] + [output.shape[-1] * output.shape[-2]]
        output = output.reshape(*transformed_shape)

        return output

    def d_out(self, dimensionality: int):
        return 2 * self.num_octaves * dimensionality
