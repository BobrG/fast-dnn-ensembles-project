"""
    AE model setup
    ...
"""
import math
import torch.nn as nn
import torch
import torchvision

import curves

class AEBase(nn.Module):
    def __init__(self, nc, ngf, ndf, bottleneck = 128):
        super(AEBase, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4, bottleneck)

        # decoder
        self.d1 = nn.Linear(bottleneck, ngf*8*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=1)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4)

        return self.fc1(h5)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))


    def forward(self, x):
        z = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        res = self.decode(z)
        return res

class AECurve(nn.Module):
    def __init__(self, nc, ngf, ndf,  bottleneck = 128, fix_points=None, num_bends=3):
        super(AECurve, self).__init__()

        self.fix_points = fix_points#[fix_start] + [False] * (num_bends - 2) + [fix_end] # weight freezing 
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf

        self.e1 = curves.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, fix_points=self.fix_points, stride=2, padding=1)
        self.bn1 = curves.BatchNorm2d(ndf, fix_points=self.fix_points)

        self.e2 = curves.Conv2d(in_channels=ndf, out_channels=2 * ndf, kernel_size=4, fix_points=self.fix_points, stride=2, padding=1)
        self.bn2 = curves.BatchNorm2d(2 * ndf, fix_points=self.fix_points)

        self.e3 = curves.Conv2d(in_channels=2 * ndf, out_channels=4 * ndf, kernel_size=4, fix_points=self.fix_points, stride=2, padding=1)
        self.bn3 = curves.BatchNorm2d(4 * ndf, fix_points=self.fix_points)

        self.e4 = curves.Conv2d(in_channels=4 * ndf, out_channels=8 * ndf, kernel_size=4, fix_points=self.fix_points, stride=2, padding=1)
        self.bn4 = curves.BatchNorm2d(8 * ndf, fix_points=self.fix_points)

        self.e5 = curves.Conv2d(in_channels=8 * ndf, out_channels=8 * ndf, kernel_size=4, fix_points=self.fix_points, stride=2, padding=1)
        self.bn5 = curves.BatchNorm2d(8 * ndf, fix_points=self.fix_points)

        self.fc1 = curves.Linear(4 * 8 * ndf, bottleneck, fix_points=self.fix_points)

        # decoder
        self.d1 = curves.Linear(bottleneck, ngf * 8 * 4, fix_points=self.fix_points)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = curves.Conv2d(in_channels=2 * ngf, out_channels=8 * ngf, kernel_size=3, fix_points=self.fix_points, stride=1)
        self.bn6 = curves.BatchNorm2d(8 * ngf, fix_points=self.fix_points, eps=1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = curves.Conv2d(in_channels=8 * ngf, out_channels=4 * ngf, kernel_size=3, fix_points=self.fix_points, stride=1)
        self.bn7 = curves.BatchNorm2d(4 * ngf, fix_points=self.fix_points, eps=1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = curves.Conv2d(in_channels=4 * ngf, out_channels=2 * ngf, kernel_size=3, fix_points=self.fix_points, stride=1)
        self.bn8 = curves.BatchNorm2d(2 * ngf, fix_points=self.fix_points, eps=1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = curves.Conv2d(in_channels=2 * ngf, out_channels=ngf, kernel_size=3, fix_points=self.fix_points, stride=1)
        self.bn9 = curves.BatchNorm2d(ngf, fix_points=self.fix_points, eps=1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=1)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = curves.Conv2d(in_channels=ngf, out_channels=nc, kernel_size=3, fix_points=self.fix_points, stride=1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


         # Initialize weights

        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.fill_(1)
                    getattr(m, 'bias_%d' % i).data.zero_()

    def encode(self, x, coeffs_t):
        h1 = self.leakyrelu(self.bn1(self.e1(x, coeffs_t), coeffs_t))
        h2 = self.leakyrelu(self.bn2(self.e2(h1, coeffs_t), coeffs_t))
        h3 = self.leakyrelu(self.bn3(self.e3(h2, coeffs_t), coeffs_t))
        h4 = self.leakyrelu(self.bn4(self.e4(h3, coeffs_t), coeffs_t))
        h5 = self.leakyrelu(self.bn5(self.e5(h4, coeffs_t), coeffs_t))
        h5 = h5.view(-1, self.ndf*8*4)
        
        return self.fc1(h5, coeffs_t)

    def decode(self, z, coeffs_t):
        h1 = self.relu(self.d1(z, coeffs_t))
        h1 = h1.view(-1, self.ngf*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)), coeffs_t), coeffs_t))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)), coeffs_t), coeffs_t))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)), coeffs_t), coeffs_t))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)), coeffs_t), coeffs_t))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5)), coeffs_t))

    def forward(self, x, coeffs_t):
        z = self.encode(x.view(-1, self.nc, self.ndf, self.ngf), coeffs_t)
        res = self.decode(z, coeffs_t)
        return res


class AE:
    base = AEBase
    curve = AECurve
    kwargs = {}
