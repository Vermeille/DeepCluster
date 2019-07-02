import torch
import torch.nn as nn
import torch.nn.functional as F

from vq import VQ


def Conv(in_ch, out_ch, ks):
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2)


def ConvBNRelu(in_ch, out_ch, ks):
    return nn.Sequential(
            Conv(in_ch, out_ch, ks),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class ResBlk(nn.Module):
    def __init__(self, ch):
        super(ResBlk, self).__init__()
        self.go = nn.Sequential(
                ConvBNRelu(ch, ch * 2, 3),
                ConvBNRelu(ch * 2, ch, 3),
            )

    def forward(self, x):
        return x + self.go(x)


class Encoder(nn.Module):
    def __init__(self, arch, hidden=64, cls=512):
        super(Encoder, self).__init__()
        layers = [
            nn.BatchNorm2d(2, affine=False),
            Conv(2, hidden, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden),
        ]

        for l in arch:
            if l == 'r':
                layers.append(ResBlk(hidden))
            elif l == 'p':
                layers.append(nn.AvgPool2d(3, 2, 1))
            elif l == 'd':
                layers.append(nn.Conv2d(hidden, hidden * 2, 1))
                hidden *= 2

        layers.append(nn.AdaptiveMaxPool2d(1))
        self.layers = nn.Sequential(*layers)
        self.to_prob = nn.Conv2d(hidden, cls, 1)

        self.sobel = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.sobel.weight.data[0, 0].copy_(
                        torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
                    )
        self.sobel.weight.data[1, 0].copy_(
                        torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                    )
        self.sobel.bias.data.zero_()
        for p in self.sobel.parameters():
            p.requires_grad = False


    def forward(self, x, ret_idx=False):
        x = self.sobel(x.mean(dim=1, keepdim=True))
        x = self.layers(x)
        return self.to_prob(x).squeeze(), x.squeeze()


class Noise(nn.Module):
    def __init__(self, ch):
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.zeros(ch, 1, 1))

    def forward(self, x):
        return x + self.a * torch.randn_like(x)


def initialize(ae):
    for m in ae.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    return ae

def baseline_64(cls):
    return initialize(Encoder('rrdprdprdprpr', cls=cls))

