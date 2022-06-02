import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolution Module
class Conv_block(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Conv_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.Lrelu2 = nn.LeakyReLU(True)
        
        # When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the input. Sliding windows that would start in the right padded region are ignored.
        self.mp = nn.MaxPool2d(kernel_size=pooling, stride=pooling, ceil_mode=True) 
        #self.bn = nn.BatchNorm2d(output_channels)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
         
      def forward(self, x):
        
        x = self.Lrelu1(self.bn1(self.conv1(x)))
        out = self.mp(self.Lrelu2(self.bn2(self.conv2(x))))
        
        return out
    
class Discriminator(nn.Module):
        def __init__(self, IsVAE=False):
            super(Encoder_module, self).__init__()
            #  input sample of size  69 × 240 (x 1) - BCHW B x 1 x 69 × 240 
            #  resized by pooling, not conv
            self.Conv_block1 = Conv_block(input_channels = 1, output_channels = 32, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block2 = Conv_block(input_channels = 32, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block3 = Conv_block(input_channels = 64, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block4 = Conv_block(input_channels = 128, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block5 = Conv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            # output latent size 3 × 8 × 256  - HWC B x 256 x 3 × 8 
            
            self.Fc1 = nn.Linear(3*8*256, 1)

        def forward(self, x):
            x = self.Conv_block1(x)
            x = self.Conv_block2(x)
            x = self.Conv_block3(x)
            x = self.Conv_block4(x)
            x = self.Conv_block5(x) # 3 × 8 × 256
            x = x.view(x.size(0), -1)
            out = self.fc1(x)
            return out

if __name__ == '__main__':
        print("##Size Check")
        
        '''print("##Encoding##")
        input = torch.randn(32, 1, 69, 240)
        print("input: ", input.shape)
        p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #p2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)
        output = p(input)
        print("output1: ", output.shape)
        output = p(output)
        print("output2: ", output.shape)
        output = p(output)
        print("output3: ", output.shape)
        output = p(output)
        print("output4: ", output.shape)
        output = p(output)
        print("output5: ", output.shape)'''
        
        
        print("##Decoding##")
        input = torch.randn(32, 32, 3, 8)
        print("input: ", input.shape)
        #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding
        m = nn.ConvTranspose2d(32, 32, 3, 2, 1)
        #m2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        # 3, 8  / 2
        m3 = nn.ConvTranspose2d(32, 32, 3, 2, 1, (1,1))
        m4 = nn.ConvTranspose2d(32, 32, 3, 2, 1, (0,1))
        output = m(input)
        print("output1: ", output.shape)
        output = m4(output)
        print("output2: ", output.shape)
        output = m3(output)
        print("output3: ", output.shape)
        output = m4(output)
        print("output4: ", output.shape)
        output = m4(output)
        print("output5: ", output.shape)
        
        '''print("##Decoding 2 same with convtrans##")
        input = torch.randn(32, 32, 3, 8)
        print("input: ", input.shape)
        m = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        output = m(input)
        print("output: ", output.shape)'''
        
        
        
        print("##Decodin up sampleing##")
        input = torch.randn(32, 32, 3, 8)
        print("input: ", input.shape)
        #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding
        up = nn.Upsample(size =(5,15), mode='nearest')
        up2 = nn.Upsample(size=(9,30), mode='nearest')
        up3 = nn.Upsample(size=(18, 60), mode='nearest')
        up4 = nn.Upsample(size=(35, 120), mode='nearest')
        up5 = nn.Upsample(size=(69, 240), mode='nearest')
        #m2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        # 3, 8  / 2

        output = up(input)
        print("output1: ", output.shape)
        output = up2(output)
        print("output2: ", output.shape)
        output = up3(output)
        print("output3: ", output.shape)
        output = up4(output)
        print("output4: ", output.shape)
        output = up5(output)
        print("output5: ", output.shape)
