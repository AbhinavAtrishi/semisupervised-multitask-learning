import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .aspp import ASPP
from .convs import stacked_conv

#############################################
#########   Semantic Segmentation   #########
#############################################
class PDLSegDecoder(nn.Module):
    def __init__(self, num_classes, in_channels=2048, decoder_channels=256, aspp_channels=256, low_level_channel=(1024, 512, 256), atrous_rates=(3,6,9), low_level_channel_project= (128, 64, 32)):
        super(PDLSegDecoder, self).__init__()
        self.decoder_channels = decoder_channels
        self.aspp_channels = aspp_channels
        self.atrous_rates = atrous_rates
        self.aspp = ASPP(in_channels, self.aspp_channels, self.atrous_rates)
        fuse_conv_base = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')
        
        # Projection Convolutions from Base Encoder
        self.proj_conv1 = nn.Conv2d(low_level_channel[0], low_level_channel_project[0], 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(low_level_channel_project[0])
        self.proj_conv2 = nn.Conv2d(low_level_channel[1], low_level_channel_project[1], 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(low_level_channel_project[1])
        self.proj_conv3 = nn.Conv2d(low_level_channel[2], low_level_channel_project[2], 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(low_level_channel_project[2])

        # Fusion Convs
        self.fuse_conv1 = fuse_conv_base(aspp_channels + low_level_channel_project[0], self.decoder_channels)
        self.fuse_conv2 = fuse_conv_base(decoder_channels + low_level_channel_project[1], self.decoder_channels)
        self.fuse_conv3 = fuse_conv_base(decoder_channels + low_level_channel_project[2], self.decoder_channels)

        # Outputs
        self.num_classes = num_classes
        self.conv_f1 = fuse_conv_base(self.decoder_channels, self.decoder_channels)
        self.conv_pred = nn.Conv2d(self.decoder_channels, self.num_classes, 1)



    def forward(self, x, res_backbone):
        x = self.aspp(x)
        upscale_1 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Project Res4
        pr1 = self.proj_conv1(res_backbone['res4'])
        pr1 = F.relu(self.proj_bn1(pr1))
        # Fuse Res4
        fs1 = torch.cat((upscale_1, pr1), 1)
        fs1 = self.fuse_conv1(fs1)

        upscale_2 = F.interpolate(fs1, scale_factor=2, mode='bilinear', align_corners=False)
        # Project Res3
        pr2 = self.proj_conv2(res_backbone['res3'])
        pr2 = F.relu(self.proj_bn2(pr2))
        # Fuse Res3
        fs2 = torch.cat((upscale_2, pr2), 1)
        fs2 = self.fuse_conv2(fs2)

        upscale_3 = F.interpolate(fs2, scale_factor=2, mode='bilinear', align_corners=False)
        # Project Res2
        pr3 = self.proj_conv3(res_backbone['res2'])
        pr3 = F.relu(self.proj_bn3(pr3))
        # Fuse Res3
        fs3 = torch.cat((upscale_3, pr3), 1)
        fs3 = self.fuse_conv3(fs3)

        outputs = {}
        outputs['features'] = fs3

        # Semantic Segmentation Output
        x = self.conv_f1(fs3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) 
        x = self.conv_pred(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        outputs['semseg'] = x

        return outputs


#############################################
#########   Instance Segmentation   #########
#############################################

class PDLInstDecoder(nn.Module):
    def __init__(self, in_channels=2048, head_channels=32, decoder_channels=128, aspp_channels=256, low_level_channel=(1024, 512, 256), atrous_rates=(3,6,9), low_level_channel_project= (64, 32, 16)):
        super(PDLInstDecoder, self).__init__()
        self.decoder_channels = decoder_channels
        self.aspp_channels = aspp_channels
        self.atrous_rates = atrous_rates
        self.aspp = ASPP(in_channels, self.aspp_channels, self.atrous_rates)
        fuse_conv_base = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')
        
        # Projection Convolutions from Base Encoder
        self.proj_conv1 = nn.Conv2d(low_level_channel[0], low_level_channel_project[0], 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(low_level_channel_project[0])
        self.proj_conv2 = nn.Conv2d(low_level_channel[1], low_level_channel_project[1], 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(low_level_channel_project[1])
        self.proj_conv3 = nn.Conv2d(low_level_channel[2], low_level_channel_project[2], 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(low_level_channel_project[2])

        # Fusion Convs
        self.fuse_conv1 = fuse_conv_base(aspp_channels + low_level_channel_project[0], self.decoder_channels)
        self.fuse_conv2 = fuse_conv_base(decoder_channels + low_level_channel_project[1], self.decoder_channels)
        self.fuse_conv3 = fuse_conv_base(decoder_channels + low_level_channel_project[2], self.decoder_channels)

        #Outputs
        self.head_channels = head_channels
        self.center_conv = fuse_conv_base(self.decoder_channels, self.head_channels)
        self.center_pred = nn.Conv2d(self.head_channels, 1, 1)
        self.offset_conv = fuse_conv_base(self.decoder_channels, self.head_channels)
        self.offset_pred = nn.Conv2d(self.head_channels, 2, 1)


    def forward(self, x, res_backbone):
        x = self.aspp(x)
        upscale_1 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Project Res4
        pr1 = self.proj_conv1(res_backbone['res4'])
        pr1 = F.relu(self.proj_bn1(pr1))
        # Fuse Res4
        fs1 = torch.cat((upscale_1, pr1), 1)
        fs1 = self.fuse_conv1(fs1)

        upscale_2 = F.interpolate(fs1, scale_factor=2, mode='bilinear', align_corners=False)
        # Project Res3
        pr2 = self.proj_conv2(res_backbone['res3'])
        pr2 = F.relu(self.proj_bn2(pr2))
        # Fuse Res3
        fs2 = torch.cat((upscale_2, pr2), 1)
        fs2 = self.fuse_conv2(fs2)

        upscale_3 = F.interpolate(fs2, scale_factor=2, mode='bilinear', align_corners=False)
        # Project Res2
        pr3 = self.proj_conv3(res_backbone['res2'])
        pr3 = F.relu(self.proj_bn3(pr3))
        # Fuse Res3
        fs3 = torch.cat((upscale_3, pr3), 1)
        fs3 = self.fuse_conv3(fs3)

        outputs = {}
        outputs['features'] = fs3

        # Instance Segmentation Output
        center = self.center_conv(fs3)
        center = F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=False)
        center = self.center_pred(center)
        center = F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=False)
        outputs['centers'] = center

        offset = self.offset_conv(fs3)
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.offset_pred(offset)
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        outputs['offsets'] = offset

        return outputs
