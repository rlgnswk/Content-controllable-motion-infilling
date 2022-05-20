import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolution Module
class Conv_block(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Conv_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Lrelu2 = nn.LeakyReLU(True)
        
        self.mp = nn.MaxPool2d(kernel_size=pooling, stride=pooling)
        #self.bn = nn.BatchNorm2d(output_channels)
      def forward(self, x):
        
        x = self.Lrelu1(self.conv1(x))
        out = self.mp(self.Lrelu2(self.conv2(x)))
        
        return out
    
class Encoder_module(nn.Module):
        def __init__(self):
            super(Encoder_module, self).__init__()
            #  input sample of size  69 × 240 (x 1) - HWC
            #  resized by pooling, not conv
            self.Conv_block1 = Conv_block(input_channels = 1, output_channels = 32, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block2 = Conv_block(input_channels = 32, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block3 = Conv_block(input_channels = 64, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block4 = Conv_block(input_channels = 128, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block5 = Conv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            # output latent size 3 × 8 × 256  - HWC
            
        def forward(self, x):
            x = self.Conv_block1(x)
            x = self.Conv_block2(x)
            x = self.Conv_block3(x)
            x = self.Conv_block4(x)
            out = self.Conv_block5(x)         
            return out

class DeConv_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1):
        super(DeConv_block, self).__init__()
        self.ConvTrans1 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # upsample
        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.ConvTrans2 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False) # no sizing
        self.Lrelu2 = nn.LeakyReLU(True)
        
        #self.BN = nn.BatchNorm2d(OutChannel)
    def forward(self, x):
        x = self.Lrelu1(self.ConvTrans1(x))
        out = self.Lrelu2(self.ConvTrans2(x))
        return out
    
    
class Decoder_module(nn.Module):
        def __init__(self, input_channels, output_channels):
            super(Decoder_module, self).__init__()
            # input latent size 3 × 8 × 256  - HWC
            self.DeConv_block1 = DeConv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1)
            self.DeConv_block2 = DeConv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1)
            self.DeConv_block3 = DeConv_block(input_channels = 256, output_channels = 128, kernel_size=3, stride=1, padding=1)
            self.DeConv_block4 = DeConv_block(input_channels = 128, output_channels = 64, kernel_size=3, stride=1, padding=1)
            #self.DeConv_block5 = DeConv_block(input_channels = output_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)
            self.ConvTrans_last2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.Lrelu = nn.LeakyReLU(True)
            self.ConvTrans_last = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
            #  output of size  69 × 240 (x 1) - HWC
        def forward(self, x):
            x = self.DeConv_block1(x)
            x = self.DeConv_block2(x)
            x = self.DeConv_block3(x)
            x = self.DeConv_block4(x)
            x = self.Lrelu(self.ConvTrans_last2(x))
            out = self.ConvTrans_last(x) # no acivation at last
            return out
        
        
class Convolutional_AE(nn.Module):
    def __init__(self):
        super(Convolutional_AE, self).__init__()
        # input sample of size 69 × 240
        self.Incoder_module = Encoder_module()
        self.Decoder_module = Decoder_module()

    def forward(self, x):
        latent = self.Incoder_module(x)
        out = self.Decoder_module(latent)
        return out