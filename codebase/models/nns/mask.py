#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software;
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import nn
from torch.nn import Linear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret


class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z2_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept

        self.elu = nn.ELU()
        # Create named nets net1..net8 for up to 8 concepts.
        # All share the same architecture; unused nets add negligible overhead.
        for idx in range(1, 9):
            setattr(self, f'net{idx}', nn.Sequential(
                nn.Linear(z2_dim, 32),
                nn.ELU(),
                nn.Linear(32, z2_dim),
            ))
        self.net = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )

    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def masked_sep(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def mix(self, z):
        """Apply a separate MLP to each concept's sub-vector."""
        zy = z.view(-1, self.concept * self.z2_dim)
        if self.z2_dim == 1:
            zy = zy.reshape(zy.size(0), zy.size(1), 1)
            parts = [zy[:, i] for i in range(self.concept)]
        else:
            parts = list(torch.split(zy, self.z2_dim, dim=1))
        outputs = [getattr(self, f'net{i + 1}')(parts[i]) for i in range(self.concept)]
        return torch.cat(outputs, dim=1)


class CausalLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z1_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ELU(),
            nn.Linear(128, z_dim),
        )

    def calculate(self, z, v):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z, v

    def masked_sep(self, z, v):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z, v

    def calculate_dag(self, z, v):
        zy = z.view(-1, self.concept * self.z1_dim)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size(0), zy.size(1), 1)
            zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        return h, v


class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M = nn.Parameter(
            torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1)
        )
        self.sigmd = torch.nn.Sigmoid()

    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmd(a)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e, A


class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features, i=False, bias=False, initial=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features, out_features)
        if initial:
            self.a[0][1], self.a[0][2], self.a[0][3] = 1, 1, 1
            self.a[1][2], self.a[1][3] = 1, 1

        self.A = nn.Parameter(self.a)

        self.b = torch.eye(out_features)
        self.B = nn.Parameter(self.b)

        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def mask_z(self, x):
        self.B = self.A
        x = torch.matmul(self.B.t(), x)
        return x

    def mask_u(self, x):
        self.B = self.A
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x

    def inv_cal(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, self.I - self.A, self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v

    def calculate_dag(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v

    def calculate_cov(self, x, v):
        v = ut.vector_expand(v)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        return x, v

    def calculate_gaussian_ini(self, x, v):
        print(self.A)
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - self.A), torch.inverse(self.I - self.A)), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()
        return x, v

    def forward(self, x):
        x = x * torch.inverse((self.A) + self.I)
        return x

    def calculate_gaussian(self, x, v):
        print(self.A)
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()
        return x, v


class ConvEncoder(nn.Module):
    def __init__(self, out_dim=None, channel=4):
        super().__init__()
        # encode() path: 96x96 -> 48 -> 24 -> 12 -> 6 (spatial)
        self.conv1 = torch.nn.Conv2d(channel, 32, 4, 2, 1)   # 48x48
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False)  # 24x24
        self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False)   # 12x12
        self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1)  # 6x6
        self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1)  # 6x6
        self.mean_layer = nn.Sequential(torch.nn.Linear(6 * 6, 16))
        self.var_layer  = nn.Sequential(torch.nn.Linear(6 * 6, 16))

        # encode_simple() path: works for 64x64 RGB input → (batch, 64, 1, 1) mean/var
        # Spatial progression for 64x64: 64->32->16->16->8->4->4->2->1->1
        self.conv6 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),       # 64→32
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),       # 32→16
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),       # 16→16 (extra depth)
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),      # 16→8
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),     # 8→4
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),     # 4→4  (extra depth)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),     # 4→2
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),     # 2→1
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 1),            # 1×1 projection → gaussian_parameters splits to 64
        )

    def encode(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.LReLU(self.conv2(x))
        x = self.LReLU(self.conv3(x))
        hm = self.convm(x)
        hm = hm.view(-1, 6 * 6)
        hv = self.convv(x)
        hv = hv.view(-1, 6 * 6)
        mu, var = self.mean_layer(hm), self.var_layer(hv)
        var = F.softplus(var) + 1e-8
        return mu, var

    def encode_simple(self, x):
        x = self.conv6(x)                       # (batch, 128, 1, 1)
        m, v = ut.gaussian_parameters(x, dim=1) # (batch, 64, 1, 1) each
        return m, v


class ConvDecoder(nn.Module):
    def __init__(self, out_dim=None, in_features=4, channel=4, image_size=64):
        super().__init__()
        self.in_features = in_features
        self.image_size = image_size

        if image_size == 64:
            # Direct 1x1 -> 64x64 with extra depth layers at 4x4 and 32x32
            self.net6 = nn.Sequential(
                nn.Conv2d(in_features, 256, 1),                   # 1→1, project up
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4),                  # 1→4
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 128, 3, 1, 1),            # 4→4 (extra depth)
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),             # 4→8
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),              # 8→16
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),              # 16→32
                nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 32, 3, 1, 1),              # 32→32 (extra depth)
                nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, channel, 4, 2, 1),         # 32→64
            )
            self._use_interpolate = False
        else:
            # 1x1 -> 128x128 native, then bilinear-interpolate to image_size (e.g. 96)
            self.net6 = nn.Sequential(
                nn.Conv2d(in_features, 128, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4),           # 1->4
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),      # 4->8
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),      # 8->16
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),      # 16->32
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),      # 32->64
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, channel, 4, 2, 1), # 64->128
            )
            self._use_interpolate = True

    def decode_sep(self, x):
        z = self.decode(x)
        return z, z, z, z, z

    def decode(self, z):
        z = z.view(-1, self.in_features, 1, 1)
        z = self.net6(z)
        if self._use_interpolate:
            z = F.interpolate(z, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)
        return z


class ConvDec(nn.Module):
    def __init__(self, out_dim=None, z_dim=16, z1_dim=4, concept=4, channel=4):
        super().__init__()
        self.concept = concept
        self.z1_dim = z1_dim
        self.z_dim = z_dim
        self.net1 = ConvDecoder(in_features=z1_dim, channel=channel, image_size=96)
        self.net2 = ConvDecoder(in_features=z1_dim, channel=channel, image_size=96)
        self.net3 = ConvDecoder(in_features=z1_dim, channel=channel, image_size=96)
        self.net4 = ConvDecoder(in_features=z1_dim, channel=channel, image_size=96)

    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        zy1, zy2, zy3, zy4 = torch.split(z, self.z1_dim, dim=1)
        rx1 = self.net1.decode(zy1)
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        out = (rx1 + rx2 + rx3 + rx4) / 4
        return out, out, out, out, out

    def decode(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        zy1, zy2, zy3, zy4 = torch.split(z, self.z1_dim, dim=1)
        rx1 = self.net1.decode(zy1)
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        return (rx1 + rx2 + rx3 + rx4) / 4


class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.channel = channel
        self.fc1 = nn.Linear(self.channel * 96 * 96, 300)
        self.fc2 = nn.Linear(300 + y_dim, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 2 * z_dim)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.net = nn.Sequential(
            nn.Linear(self.channel * 96 * 96, 900),
            nn.ELU(),
            nn.Linear(900, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def conditional_encode(self, x, l):
        x = x.view(-1, self.channel * 96 * 96)
        x = F.elu(self.fc1(x))
        l = l.view(-1, 4)
        x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        m, v = ut.gaussian_parameters(x, dim=1)
        return m, v

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        xy = xy.view(-1, self.channel * 96 * 96)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v


class Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel=4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel
        self.elu = nn.ELU()

        # Concept-specific decoders (net1..net4 + net5_c..net8_c for up to 8 concepts)
        def _make_concept_net():
            return nn.Sequential(
                nn.Linear(z1_dim + y_dim, 300),
                nn.ELU(),
                nn.Linear(300, 300),
                nn.ELU(),
                nn.Linear(300, 1024),
                nn.ELU(),
                nn.Linear(1024, self.channel * 96 * 96),
            )

        self.net1 = _make_concept_net()
        self.net2 = _make_concept_net()
        self.net3 = _make_concept_net()
        self.net4 = _make_concept_net()
        self.net5_c = _make_concept_net()
        self.net6_c = _make_concept_net()
        self.net7_c = _make_concept_net()
        self.net8_c = _make_concept_net()

        # Auxiliary nets (kept for decode_union and decode)
        self.net5 = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96),
        )
        self.net6 = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96),
        )

    @property
    def _concept_nets(self):
        return [self.net1, self.net2, self.net3, self.net4,
                self.net5_c, self.net6_c, self.net7_c, self.net8_c]

    def decode_union(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size(0), zy.size(1), 1)
            parts = [zy[:, i] for i in range(self.concept)]
        else:
            parts = list(torch.split(zy, self.z_dim // self.concept, dim=1))
        nets = self._concept_nets
        outputs = [nets[i](parts[i]) for i in range(self.concept)]
        h = self.net5(sum(outputs) / self.concept)
        return h, h, h, h, h

    def decode(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        h = self.net6(z)
        return h, h, h, h, h

    def decode_sep(self, z, u, y=None):
        """Dynamically handles any concept count from 1 to 8."""
        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)

        if self.z1_dim == 1:
            zy = zy.reshape(zy.size(0), zy.size(1), 1)
            parts = [zy[:, i] for i in range(self.concept)]
        else:
            parts = list(torch.split(zy, self.z1_dim, dim=1))

        nets = self._concept_nets
        outputs = [nets[i](parts[i]) for i in range(self.concept)]
        h = sum(outputs) / self.concept
        return h, h, h, h, h

    def decode_mix(self, z):
        z = z.permute(0, 2, 1)
        z = torch.sum(z, dim=2, out=None)
        z = z.contiguous()
        h = self.net1(z)
        return h

    def decode_condition(self, z, u):
        z = z.view(-1, 3 * 4)
        z1, z2, z3 = torch.split(z, self.z_dim // 4, dim=1)
        rx1 = self.net1(torch.transpose(
            torch.cat((torch.transpose(z1, 1, 0), u[:, 0].reshape(1, u.size(0))), dim=0), 1, 0))
        rx2 = self.net2(torch.transpose(
            torch.cat((torch.transpose(z2, 1, 0), u[:, 1].reshape(1, u.size(0))), dim=0), 1, 0))
        rx3 = self.net3(torch.transpose(
            torch.cat((torch.transpose(z3, 1, 0), u[:, 2].reshape(1, u.size(0))), dim=0), 1, 0))
        h = self.net4(torch.cat((rx1, rx2, rx3), dim=1))
        return h

    def decode_cat(self, z, u, y=None):
        z = z.view(-1, 4 * 4)
        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5(torch.cat((rx1, rx2, rx3, rx4), dim=1))
        return h


class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 4 * 96 * 96),
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)


class Classifier(nn.Module):
    def __init__(self, y_dim):
        super().__init__()
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim),
        )

    def classify(self, x):
        return self.net(x)


# ── UNet encoder / decoder for CausalVAE ─────────────────────────────────────

class DoubleConv(nn.Module):
    """Two rounds of Conv2d → GroupNorm → LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetEncoderVAE(nn.Module):
    """
    UNet-style encoder for CausalVAE.

    Produces:
      feat  — (batch, 128, 1, 1)  fed into gaussian_parameters() → 64 mean + 64 var
              (same interface as ConvEncoder.encode_simple)
      skips — (s1, s2, s3, s4, b)  feature maps at each scale for the decoder

    Spatial progression for 64×64 RGB input:
      64×64 (32ch) → 32×32 (64ch) → 16×16 (128ch) → 8×8 (256ch) → 4×4 (256ch) → 1×1 (128ch)
    """
    def __init__(self, channel: int = 3):
        super().__init__()
        self.enc1       = DoubleConv(channel, 32)
        self.enc2       = DoubleConv(32,  64)
        self.enc3       = DoubleConv(64,  128)
        self.enc4       = DoubleConv(128, 256)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 256)          # 4×4, 256 ch
        self.proj       = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                    # 4×4 → 1×1
            nn.Conv2d(256, 128, 1),                     # 256 → 128 ch (split to 64+64)
        )

    def encode(self, x: torch.Tensor):
        """
        Returns:
            feat  (batch, 128, 1, 1)  — passed to gaussian_parameters
            skips (s1, s2, s3, s4, b) — encoder feature maps for decoder
        """
        s1 = self.enc1(x)                    # 64×64, 32 ch
        s2 = self.enc2(self.pool(s1))        # 32×32, 64 ch
        s3 = self.enc3(self.pool(s2))        # 16×16, 128 ch
        s4 = self.enc4(self.pool(s3))        #  8×8,  256 ch
        b  = self.bottleneck(self.pool(s4))  #  4×4,  256 ch
        feat = self.proj(b)                  #  1×1,  128 ch
        return feat, (s1, s2, s3, s4, b)


class UNetDecoderVAE(nn.Module):
    """
    UNet-style decoder for CausalVAE.

    Accepts z_4d (batch, z_dim, 1, 1) + skip feature maps from UNetEncoderVAE.
    Uses bilinear upsampling + skip-concat at every scale for sharp reconstructions.

    Channel flow:
      z(z_dim,1×1) → expand to (256,4×4)
      dec4: cat(256, b=256) → DoubleConv(512→256)   4×4
      dec3: cat(256, s4=256)→ DoubleConv(512→128)   8×8
      dec2: cat(128, s3=128)→ DoubleConv(256→64)   16×16
      dec1: cat(64,  s2=64) → DoubleConv(128→32)   32×32
      dec0: cat(32,  s1=32) → DoubleConv(64→32)    64×64
      final: Conv2d(32→channel, 1×1)
    """
    def __init__(self, in_features: int = 32, channel: int = 3):
        super().__init__()
        self.in_features = in_features
        # Expand z from (z_dim,1,1) to (256,4,4)
        self.from_z = nn.Sequential(
            nn.Conv2d(in_features, 256, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(size=4, mode='bilinear', align_corners=False),
        )
        self.dec4  = DoubleConv(256 + 256, 256)
        self.dec3  = DoubleConv(256 + 256, 128)
        self.dec2  = DoubleConv(128 + 128, 64)
        self.dec1  = DoubleConv(64  + 64,  32)
        self.dec0  = DoubleConv(32  + 32,  32)
        self.final = nn.Conv2d(32, channel, 1)

    def decode(self, z_4d: torch.Tensor, skips) -> torch.Tensor:
        """
        Args:
            z_4d:  (batch, z_dim, 1, 1)
            skips: (s1, s2, s3, s4, b) from UNetEncoderVAE.encode()
        Returns:
            (batch, channel, 64, 64)
        """
        s1, s2, s3, s4, b = skips

        def up(t):
            return F.interpolate(t, scale_factor=2,
                                 mode='bilinear', align_corners=False)

        h = self.from_z(z_4d)                          # 4×4, 256
        h = self.dec4(torch.cat([h,    b ], dim=1))    # 4×4, 256
        h = self.dec3(torch.cat([up(h), s4], dim=1))   # 8×8, 128
        h = self.dec2(torch.cat([up(h), s3], dim=1))   # 16×16, 64
        h = self.dec1(torch.cat([up(h), s2], dim=1))   # 32×32, 32
        h = self.dec0(torch.cat([up(h), s1], dim=1))   # 64×64, 32
        return self.final(h)


