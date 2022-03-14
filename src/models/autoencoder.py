class AE(nn.Module):
    def __init__(self, num_channels, encoder_features, decoder_features, bottleneck = 128):
        super(AE, self).__init__()

        # encoder
        self.encover_blocks.append(self.conv_block(num_channels, encoder_features))
        encoder_levels = [1, 2, 4, 8, 8]
        for i, step in enumerate(encoder_levels[1:]):
            self.encoder_blocks.append(self.conv_block(levels[i-1]*encoder_features, step*encoder_features))
        
        #self.e1 = nn.Conv2d(num_channels, encoder_features, 4, 2, 1)
        #self.bn1 = nn.BatchNorm2d(encoder_features)

        #self.e2 = nn.Conv2d(encoder_features, encoder_features*2, 4, 2, 1)
        #self.bn2 = nn.BatchNorm2d(encoder_features*2)

        #self.e3 = nn.Conv2d(encoder_features*2, encoder_features*4, 4, 2, 1)
        #self.bn3 = nn.BatchNorm2d(encoder_features*4)

        #self.e4 = nn.Conv2d(encoder_features*4, encoder_features*8, 4, 2, 1)
        #self.bn4 = nn.BatchNorm2d(ndf*8)

        #self.e5 = nn.Conv2d(encoder_features*8, encoder_features*8, 4, 2, 1)
        #self.bn5 = nn.BatchNorm2d(encoder_features*8)

        self.fc1 = nn.Linear(encoder_features*8*4*4, bottleneck)

        # decoder
        self.d1 = nn.Linear(bottleneck, decoder_features*8*2*4*4)

        decoder_levels = [8*2, 8, 4, 2, 1]

        #self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        #self.pd1 = nn.ReplicationPad2d(1)
        #self.d2 = nn.Conv2d(decoder_features*8*2, decoder_features*8, 3, 1)
        #self.bn6 = nn.BatchNorm2d(decoder_features*8, 1.e-3)

        #self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        #self.pd2 = nn.ReplicationPad2d(1)
        #self.d3 = nn.Conv2d(decoder_features*8, decoder_features*4, 3, 1)
        #self.bn7 = nn.BatchNorm2d(decoder_features*4, 1.e-3)

        #self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        #self.pd3 = nn.ReplicationPad2d(1)
        #self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        #self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        #self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        #self.pd4 = nn.ReplicationPad2d(1)
        #self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        #self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)
        for i, step in enumerate(decoder_levels[:-1]):
            self.decoder_blocks.append(nn.BatchNorm2d(self.deconv_block(step*decoder_features, levels[i+1]*decoder_features), 1.e-3))

        self.decoder_blocks.append(self.deconv_block(decoder_features, num_channels))
        
        #self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        #self.pd5 = nn.ReplicationPad2d(1)
        #self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.nef = encoder_features
        self.ndf = decoder_features
        self.nc = num_channels

    def conv_block(in_feat, out_feat):

        return nn.Sequential(nn.Conv2d(in_feat, out_feat, 4, 2, 1), 
                             nn.BatchNorm2d(out_feat))

    def deconv_block(in_feat, out_feat):

        return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                             nn.ReplicationPad2d(1),
                             nn.Conv2d(in_feat, out_feat, 3, 1))

    def encode(self, x):

        encoder_out = self.leaky_relu(self.encoder_blocks[0](x))
        for block in self.encoder_blocks:
            encoder_out = self.leakyrelu(block(encoder_out))

        return self.fc1(encoder_out.view(-1, self.nef*8*4*4))
        #h1 = self.leakyrelu(self.bn1(self.e1(x)))
        #h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        #h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        #h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        #h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        #h5 = h5.view(-1, self.ndf*8*4*4)

        #return self.fc1(h5)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        decoder_out = h1.view(-1, self.ndf*8*2, 4, 4)
        for block in self.decoder_blocks:
            decoder_out = self.leakyrelu(self.decoder_blocks(decoder_out))

        #h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        #h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        #h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        #h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        #return self.sigmoid(self.d6(self.pd5(self.up5(h5))))
        return self.sigmoid(decoder_out)

    def forward(self, x):
        z = self.encode(x.view(-1, self.nc, self.nef, self.ndf))
        res = self.decode(z)
        return res
