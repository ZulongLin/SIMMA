import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in,
                               out_channels=64,
                               kernel_size=3,
                               padding=1,
                               padding_mode='circular')
        self.norm1 = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True, device=None, dtype=None)
        self.activate = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               padding=1,
                               padding_mode='circular')
        self.norm2 = nn.LayerNorm(32, eps=1e-5, elementwise_affine=True, device=None, dtype=None)
        nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               padding=1,
                               padding_mode='circular')
        self.norm3 = nn.LayerNorm(16, eps=1e-5, elementwise_affine=True, device=None, dtype=None)
        nn.ELU()
        self.conv4 = nn.Conv1d(in_channels=16,
                               out_channels=1,
                               kernel_size=3,
                               padding=1,
                               padding_mode='circular')

    def forward(self, x):
        t1 = self.activate(self.norm1(self.conv1(x).permute(0, 2, 1)))
        t2 = self.activate(self.norm2(self.conv2(t1.permute(0, 2, 1)).permute(0, 2, 1)))
        t3 = self.activate(self.norm3(self.conv3(t2.permute(0, 2, 1)).permute(0, 2, 1)))
        down_result = self.conv4(t3.permute(0, 2, 1)).permute(0, 2, 1)
        # down_result = self.downModule(x)
        return down_result


class SpatialCompressionBlock(nn.Module):
    def __init__(self, a=0, b=0, c=0):
        super(SpatialCompressionBlock, self).__init__()
        self.a = a
        self.b = b
        self.c = c

        self.conv1 = ConvLayer(c_in=a) if a > 0 else None
        self.conv2 = ConvLayer(c_in=b) if b > 0 else None
        self.conv3 = ConvLayer(c_in=c) if c > 0 else None

    def forward(self, x):
        # The input x has a shape of (T, D), where D = a + b + c

        # Slice the input according to a, b, and c
        x_a = x[:, :, :self.a].permute(0, 2, 1) if self.a > 0 else None
        x_b = x[:, :, self.a:self.a + self.b].permute(0, 2, 1) if self.b > 0 else None
        x_c = x[:, :, self.b:].permute(0, 2, 1) if self.c > 0 else None

        # Compress using convolution
        compressed_a = self.conv1(x_a) if self.conv1 else None
        compressed_b = self.conv2(x_b) if self.conv2 else None
        compressed_c = self.conv3(x_c) if self.conv3 else None

        # Concatenate the compressed results
        compressed_list = [compressed_a, compressed_b, compressed_c]
        compressed = torch.cat([x for x in compressed_list if x is not None], dim=1)
        return compressed


class split_modals_features(nn.Module):
    def __init__(self, a=0, b=0, c=0):
        super(split_modals_features, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        # The input x has a shape of (T, D), where D = a + b + c

        # Slice the input according to a, b, and c
        x_a = x[:, :, :self.a] if self.a > 0 else None
        x_b = x[:, :, self.a:self.a + self.b] if self.b > 0 else None
        x_c = x[:, :, self.a + self.b:] if self.c > 0 else None

        return x_a, x_b, x_c
