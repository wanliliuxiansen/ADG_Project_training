import functools

import torch.nn as nn
import torch
import torchvision.models as models
'''
生成器
'''
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        #128 124
        self.layer_0_0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        #124 120
        self.layer_0_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        #120 116
        self.layer_0_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        #116 112
        self.layer_0 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        #112 108
        self.layer_1_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        #108 104
        self.layer_1_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        #104 98 128
        self.layer_1_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.Upsample(size=[128,128], mode='bilinear'),
            nn.LeakyReLU(0.2, True)
        )
    def cut_and(self, x_1, x_2, index):
        batch_size,deep,h,w = list(x_2.size())
        x_1 = nn.functional.interpolate(x_1, size=[h,w], mode='bilinear')
        return torch.cat((x_1, x_2), index)
    def forward(self, x):

        x_0_0 = self.layer_0_0(x)
        x_0_1 = self.layer_0_1(x_0_0)
        x_0_2 = self.layer_0_2(x_0_1)

        x_0 = self.layer_0(x_0_2)
        x_0 = self.cut_and(x_0_2, x_0 ,1)

        x_1_0 = self.layer_1_0(x_0)
        x_1_0 = self.cut_and(x_0_1, x_1_0, 1)

        x_1_1 = self.layer_1_1(x_1_0)
        x_1_1 = self.cut_and(x_0_0, x_1_1, 1)

        x_1_2 = self.layer_1_2(x_1_1)
        return x_1_2



'''
判别器
'''
class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # print(input.shape)
        return self.model(input)