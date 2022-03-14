import torch
import torch.nn as nn
import numpy as np

class AE(nn.Module):
    def __init__(self, num_channels, encoder_features, decoder_features, bottleneck = 128):
        super(AE, self).__init__()
        self.encoder_blocks = []
        self.decoder_blocks = []

        # encoder
        self.encoder_blocks += self.conv_block(num_channels, encoder_features)
        
        encoder_levels = [1, 2, 4, 8, 8]
        for i, step in enumerate(encoder_levels[1:]):
            self.encoder_blocks += self.conv_block(encoder_levels[i-1]*encoder_features, step*encoder_features)
        
        selt.encoder

        self.fc1 = nn.Linear(encoder_features*8*4*4, bottleneck)

        # decoder
        self.d1 = nn.Linear(bottleneck, decoder_features*8*2*4*4)

        decoder_levels = [8*2, 8, 4, 2, 1]
        for i, step in enumerate(decoder_levels[:-1]):
            self.decoder_blocks.append(self.deconv_block(step*decoder_features, decoder_levels[i+1]*decoder_features))

        self.decoder_blocks.append(self.deconv_block(decoder_features, num_channels, batchnorm=False))
        
        # activations 
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.nef = encoder_features
        self.ndf = decoder_features
        self.nc = num_channels

    def conv_block(self, in_feat, out_feat):

        return nn.Sequential(nn.Conv2d(in_feat, out_feat, 4, 2, 1), 
                             nn.BatchNorm2d(out_feat))

    def deconv_block(self, in_feat, out_feat, batchnorm=True):

        if batchnorm:
            return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(in_feat, out_feat, 3, 1), 
                                 nn.BatchNorm2d(out_feat, 1.e-3))
        else:
            return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(in_feat, out_feat, 3, 1))

    def encode(self, x):

        encoder_out = self.leakyrelu(self.encoder_blocks[0](x))
        for block in self.encoder_blocks:
            encoder_out = self.leakyrelu(block(encoder_out))

        return self.fc1(encoder_out.view(-1, self.nef*8*4*4))

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        decoder_out = h1.view(-1, self.ndf*8*2, 4, 4)
        for block in self.decoder_blocks:
            decoder_out = self.leakyrelu(self.decoder_blocks(decoder_out))

        return self.sigmoid(decoder_out)

    def forward(self, x):
        z = self.encode(x.view(-1, self.nc, self.nef, self.ndf))
        res = self.decode(z)
        return res
