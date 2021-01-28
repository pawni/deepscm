import torch
from torch import nn

import numpy as np


def get_ops(norm_type: str = 'batch', n_dim: int = 2):
    if norm_type == 'batch':
        bn_1d_op = nn.BatchNorm1d
    elif norm_type == 'instance':
        bn_1d_op = nn.InstanceNorm1d
    elif norm_type == 'layer':
        bn_1d_op = nn.LayerNorm
    elif 'none' in norm_type:
        bn_1d_op = nn.Identity

    if n_dim == 2:
        conv_op = nn.Conv2d
        conv_transpose_op = nn.ConvTranspose2d
        if 'instance' in norm_type:
            bn_op = nn.InstanceNorm2d
        elif 'batch' in norm_type:
            bn_op = nn.BatchNorm2d
        else:
            bn_op = nn.Identity
    elif n_dim == 3:
        conv_op = nn.Conv3d
        conv_transpose_op = nn.ConvTranspose3d
        if 'instance' in norm_type:
            bn_op = nn.InstanceNorm3d
        elif 'batch' in norm_type:
            bn_op = nn.BatchNorm3d
        else:
            bn_op = nn.Identity

    return bn_1d_op, bn_op, conv_op, conv_transpose_op


class Encoder(nn.Module):
    def __init__(self, num_convolutions: int = 1, filters=(16, 32, 64, 128), latent_dim: int = 128, input_size=(1, 192, 192), norm_type: str = 'batch'):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim

        assert len(input_size) in [3, 4]
        self.n_dim = len(input_size) - 1

        bn_1d_op, bn_op, conv_op, _ = get_ops(norm_type, self.n_dim)

        layers = []

        cur_channels = 1
        for c in filters:
            for _ in range(0, num_convolutions - 1):
                layers += [conv_op(cur_channels, c, 3, 1, 1), bn_op(c), nn.LeakyReLU(.1, inplace=True)]
                cur_channels = c

            layers += [conv_op(cur_channels, c, 4, 2, 1), bn_op(c), nn.LeakyReLU(.1, inplace=True)]

            cur_channels = c

        self.cnn = nn.Sequential(*layers)

        self.intermediate_shape = np.array(input_size) // (2 ** len(filters))
        self.intermediate_shape[0] = cur_channels

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shape), latent_dim),
            bn_1d_op(latent_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn(x).view(-1, np.prod(self.intermediate_shape))

        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, num_convolutions=1, filters=(128, 64, 32, 16), latent_dim: int = 128, output_size=(1, 192, 192),
                 upconv=False, n_dim: int = 2, norm_type: str = 'batch'):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim

        assert len(output_size) in [3, 4]
        self.n_dim = len(output_size) - 1

        bn_1d_op, bn_op, conv_op, conv_transpose_op = get_ops(norm_type, self.n_dim)

        self.intermediate_shape = np.array(output_size) // (2 ** (len(filters) - 1))
        self.intermediate_shape[0] = filters[0]

        self.fc = nn.Sequential(
            nn.Conv1d(latent_dim, np.prod(self.intermediate_shape), 1, 1, 0),
            bn_1d_op(np.prod(self.intermediate_shape)),
            nn.LeakyReLU(.1, inplace=True)
        )

        layers = []

        cur_channels = filters[0]
        for c in filters[1:]:
            for _ in range(0, num_convolutions - 1):
                layers += [conv_op(cur_channels, cur_channels, 3, 1, 1), bn_op(cur_channels), nn.LeakyReLU(.1, inplace=True)]

            if upconv:
                layers += [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    conv_op(cur_channels, c, kernel_size=5, stride=1, padding=2)
                ]
            else:
                layers += [conv_transpose_op(cur_channels, c, kernel_size=4, stride=2, padding=1)]
            layers += [bn_op(c), nn.LeakyReLU(.1, inplace=True)]

            cur_channels = c

        layers += [conv_op(cur_channels, 1, 1, 1)]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.fc(x.view(-1, self.latent_dim, 1)).view(-1, *self.intermediate_shape)

        return self.cnn(x)
