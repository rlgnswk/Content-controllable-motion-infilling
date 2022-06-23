import torch
import torch.nn as nn
import torch.nn.functional as F
# Convolution Module
class Conv_block_IN(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Conv_block_IN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        
        self.in1 = nn.InstanceNorm2d(output_channels)

        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.in2 = nn.InstanceNorm2d(output_channels)
        
        self.Lrelu2 = nn.LeakyReLU(True)
        
        # When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the input. Sliding windows that would start in the right padded region are ignored.
        self.mp = nn.MaxPool2d(kernel_size=pooling, stride=pooling, ceil_mode=True) 
        #self.bn = nn.BatchNorm2d(output_channels)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
         
      def forward(self, x):
        
        x = self.Lrelu1(self.in1(self.conv1(x)))
        out = self.mp(self.Lrelu2(self.in2(self.conv2(x))))
        
        return out
    
class Encoder_module_IN(nn.Module):
        def __init__(self, IsVAE=False):
            super(Encoder_module_IN, self).__init__()
            #  input sample of size  69 × 240 (x 1) - BCHW B x 1 x 69 × 240 
            #  resized by pooling, not conv
            self.Conv_block1 = Conv_block_IN(input_channels = 1, output_channels = 32, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block2 = Conv_block_IN(input_channels = 32, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block3 = Conv_block_IN(input_channels = 64, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block4 = Conv_block_IN(input_channels = 128, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block5 = Conv_block_IN(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            # output latent size 3 × 8 × 256  - HWC B x 256 x 3 × 8 
        def forward(self, x):
            x = self.Conv_block1(x)
            x = self.Conv_block2(x)
            x = self.Conv_block3(x)
            x = self.Conv_block4(x)
            out = self.Conv_block5(x)
            return out


class DeConv_block_upsampling(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, size=None):
        super(DeConv_block_upsampling, self).__init__()
        self.up = nn.Upsample(size=size, mode='nearest')
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        #self.ConvTrans1 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding) # upsample
        self.gn1 = nn.GroupNorm(output_channels, output_channels)
        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        #self.ConvTrans2 = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding) # no sizing
        self.gn2 = nn.GroupNorm(output_channels, output_channels)
        self.Lrelu2 = nn.LeakyReLU(True)
        
        
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        #self.BN = nn.BatchNorm2d(OutChannel)
    def forward(self, x):
        x = self.up(x)
        x = self.Lrelu1(self.gn1(self.conv1(x)))
        #x = self.Lrelu1(self.ConvTrans1(x))
        out = self.Lrelu2(self.gn2(self.conv2(x)))
        #out = self.Lrelu2(self.ConvTrans2(x))
        return out

class Decoder_module_upsampling(nn.Module):
        def __init__(self):
            super(Decoder_module_upsampling, self).__init__()
            # input latent size 3 × 8 × 256  - HWC
            self.DeConv_block1 = DeConv_block_upsampling(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, size=(5,15))
            self.DeConv_block2 = DeConv_block_upsampling(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, size=(9,30))
            self.DeConv_block3 = DeConv_block_upsampling(input_channels = 256, output_channels = 128, kernel_size=3, stride=1, padding=1, size=(18, 60))
            self.DeConv_block4 = DeConv_block_upsampling(input_channels = 128, output_channels = 64, kernel_size=3, stride=1, padding=1, size=(35, 120))
            #self.DeConv_block5 = DeConv_block(input_channels = output_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)
            self.up = nn.Upsample(size=(69, 240), mode='nearest')
            self.Conv_last2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
            self.Lrelu = nn.LeakyReLU(True)
            self.Conv_last = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

            nn.init.kaiming_uniform_(self.Conv_last2.weight)
            nn.init.kaiming_uniform_(self.Conv_last.weight)

            #  output of size  69 × 240 (x 1) - HWC
        def forward(self, x):
            x = self.DeConv_block1(x)
            x = self.DeConv_block2(x)
            x = self.DeConv_block3(x)
            x = self.DeConv_block4(x)
            x = self.up(x)
            x = self.Lrelu(self.Conv_last2(x))
            out = self.Conv_last(x) # no acivation at last
            return out

# Convolution Module
class Conv_block_GroupNorm(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Conv_block_GroupNorm, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.gn1 = nn.GroupNorm(output_channels,output_channels)

        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.gn2 = nn.GroupNorm(output_channels,output_channels)
        
        self.Lrelu2 = nn.LeakyReLU(True)
        
        # When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the input. Sliding windows that would start in the right padded region are ignored.
        self.mp = nn.MaxPool2d(kernel_size=pooling, stride=pooling, ceil_mode=True) 
        #self.bn = nn.BatchNorm2d(output_channels)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
         
      def forward(self, x):
        
        x = self.Lrelu1(self.gn1(self.conv1(x)))
        #out = self.mp(self.Lrelu2(self.gn2(self.conv2(x))))
        out = self.Lrelu2(self.gn2(self.conv2(x)))
        
        return out
    
class Style_Encoder(nn.Module):
        def __init__(self, IsVAE=False):
            super(Style_Encoder, self).__init__()
            #  input sample of size  69 × 30 (x 1) - BCHW B x 1 x 69 × 240 
            #  resized by pooling, not conv
            self.Conv_block1 = Conv_block_GroupNorm(input_channels = 1, output_channels = 32, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block2 = Conv_block_GroupNorm(input_channels = 32, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block3 = Conv_block_GroupNorm(input_channels = 64, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block4 = Conv_block_GroupNorm(input_channels = 128, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            #self.Conv_block5 = Conv_block_GroupNorm(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            # output latent size 3 × 8 × 256  - HWC B x 256 x 3 × 8 
            
            self.Fc_mean = nn.Linear(69*1*128, 256)
            self.Fc_std = nn.Linear(69*1*128, 256)

        def forward(self, x):
            x = self.Conv_block1(x)
            x = self.Conv_block2(x)
            x = self.Conv_block3(x)
            x = self.Conv_block4(x)
            #print(x.shape)
            #x = self.Conv_block5(x) # 256 x 3 × 8 
            x = x.view(x.size(0), -1)
            mean = self.Fc_mean(x)
            std = self.Fc_std(x)
            
            return mean, std

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def AdaIN(content_feat, style_mean, style_std):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    #style_mean, style_std = self.calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    
    return normalized_feat * style_std.unsqueeze(-1).unsqueeze(-1).expand(size) + style_mean.unsqueeze(-1).unsqueeze(-1).expand(size)

class Convolutional_blend(nn.Module):
    def __init__(self):
        super(Convolutional_blend, self).__init__()
        # input sample of size 69 × 240
        self.Content_Encoder_module = Encoder_module_IN()
        self.Style_Encoder_module = Style_Encoder()

        self.Decoder_module = Decoder_module_upsampling()
        self.blend_mean = 0.0
        self.blend_std = 0.0

    def forward(self, masked_input, blend_gt):
        mask_feat = self.Content_Encoder_module(masked_input) # 
        
        blend_mean, blend_std = self.Style_Encoder_module(blend_gt) #mean and var
        self.blend_mean = blend_mean
        self.blend_std = blend_std        
        AdaIN_latent = AdaIN(mask_feat, blend_mean, blend_std)
        
        out_affine = self.Decoder_module(AdaIN_latent)

        out_recon = self.Decoder_module(mask_feat)        
        return out_affine, out_recon


    def forward_content_encoder(self, blend_gt):

        blend_mean, blend_std = self.Style_Encoder_module(blend_gt) #mean and var
    
        return blend_mean, blend_std

    def test(self, masked_input, blend_gt, alpha):
        # make scalable output with a = [0.0, 0.1, ..., 1.0]
        # latent =  (1-a) * mask_feat + a * AdaIN_latent

        mask_feat = self.Content_Encoder_module(masked_input) # 
        
        blend_mean, blend_std = self.Style_Encoder_module(blend_gt) #mean and var
        
        AdaIN_latent = AdaIN(mask_feat, blend_mean, blend_std)

        target_latent =  (1-alpha) * mask_feat + alpha * AdaIN_latent

        out_test = self.Decoder_module(target_latent)  

        return out_test

    def test2(self, masked_input, mask_gt, blend_gt, alpha):
        # make scalable output with a = [0.0, 0.1, ..., 1.0]
        # latent =  (1-a) * mask_feat + a * AdaIN_latent

        mask_feat = self.Content_Encoder_module(masked_input) # 
        
        blend_mean, blend_std = self.Style_Encoder_module(blend_gt) #mean and var
        
        gt_mean, gt_std = self.Style_Encoder_module(mask_gt)
        
        AdaIN_latent_blend = AdaIN(mask_feat, blend_mean, blend_std)

        AdaIN_latent_gt = AdaIN(mask_feat, gt_mean, gt_std)

        target_latent =  (1-alpha) * AdaIN_latent_gt + alpha * AdaIN_latent_blend

        out_test = self.Decoder_module(target_latent)  

        return out_test

    def test_rand_mu_var(self, masked_input, batch_size = 80):
        # fixed mean and std for checking separation recon and content space 

        mask_feat = self.Content_Encoder_module(masked_input) # 
        blend_mean = torch.rand(batch_size, 256) * 2 - 1 
        blend_std = torch.rand(batch_size, 256) * 2 - 1
        AdaIN_latent_blend = AdaIN(mask_feat, blend_mean, blend_std)

        out_test = self.Decoder_module(AdaIN_latent_blend)  

        return out_test

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
            super(Discriminator, self).__init__()
            #  input sample of size  69 × 240 (x 1) - BCHW B x 1 x 69 × 240 
            #  resized by pooling, not conv
            self.Conv_block1 = Conv_block(input_channels = 1, output_channels = 32, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block2 = Conv_block(input_channels = 32, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block3 = Conv_block(input_channels = 64, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block4 = Conv_block(input_channels = 128, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block5 = Conv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            # output latent size 3 × 8 × 256  - HWC B x 256 x 3 × 8 
            
            self.Fc1 = nn.Linear(3*8*256, 1)
            self.sigmoid_layer = nn.Sigmoid()
        def forward(self, x):
            x = self.Conv_block1(x)
            x = self.Conv_block2(x)
            x = self.Conv_block3(x)
            x = self.Conv_block4(x)
            x = self.Conv_block5(x) # 3 × 8 × 256
            x = x.view(x.size(0), -1)
            out = self.Fc1(x)
            return self.sigmoid_layer(out)
    
    

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
