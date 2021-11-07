import torch
import torch.nn as nn
import torch.nn.functional as F

class TTNInst_Seg2Dep(nn.Module):
    def __init__(self, in_channels_inst, in_channels_seg):
        super(TTNInst_Seg2Dep, self).__init__()
        # Input 1 Instance Features
        self.c1i = nn.Conv2d(in_channels_inst, 64, kernel_size=3, stride=1, padding=1)
        self.c2i = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1i = nn.BatchNorm2d(128)
        # Input 2 Semseg
        self.c1s = nn.Conv2d(in_channels_seg, 64, kernel_size=3, stride=1, padding=1)
        self.c2s = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1s = nn.BatchNorm2d(128)

        self.c2m = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.c3m = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512) 
        self.c4m = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.en_depth = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.c5m = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.c6m = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.c7m = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.c8m = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.c9 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, inst_dict, semseg):
        inst_feats = F.interpolate(inst_dict['features'], scale_factor=4, mode='bilinear', align_corners=False)
        i = self.c1i(inst_feats)
        i = self.c2i(i)
        i = F.leaky_relu(self.bn1i(i))
        
        s = self.c1s(semseg)
        s = self.c2s(s)
        s = F.leaky_relu(self.bn1s(s))

        x = torch.cat([i, s], dim=1)

        x = self.c2m(x)
        x = self.c3m(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.c4m(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.en_depth(x)

        x = self.c5m(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.c6m(x)
        x = F.leaky_relu(self.bn5(x))
        x = self.c7m(x)
        x = F.leaky_relu(self.bn6(x))
        x = self.c8m(x)
        x = F.leaky_relu(self.bn7(x))

        x = self.c9(x)

        return x

class TTNInst_Dep2Seg(nn.Module):
    def __init__(self, inp_channels, n_classes = 20):
        super(TTNInst_Dep2Seg, self).__init__()
        # Input 1 Instance Features
        self.c1i = nn.Conv2d(inp_channels, 64, kernel_size=3, stride=1, padding=1)
        self.c2i = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1i = nn.BatchNorm2d(128)
        # Input 2 Depth
        self.c1d = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.c2d = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1d = nn.BatchNorm2d(128)

        self.c2m = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.c3m = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512) 
        self.c4m = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.en_depth = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.c5m = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.c6m = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.c7m = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.c8m = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.c9 = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, inst_dict, depth):
        inst_feats = F.interpolate(inst_dict['features'], scale_factor=4, mode='bilinear', align_corners=False)
        i = self.c1i(inst_feats)
        i = self.c2i(i)
        i = F.leaky_relu(self.bn1i(i))
        
        d = self.c1d(depth)
        d = self.c2d(d)
        d = F.leaky_relu(self.bn1d(d))

        x = torch.cat([i, d], dim=1)

        x = self.c2m(x)
        x = self.c3m(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.c4m(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.en_depth(x)

        x = self.c5m(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.c6m(x)
        x = F.leaky_relu(self.bn5(x))
        x = self.c7m(x)
        x = F.leaky_relu(self.bn6(x))
        x = self.c8m(x)
        x = F.leaky_relu(self.bn7(x))

        x = self.c9(x)
        return x

class TTNSeg_Dep2Inst(nn.Module):
    def __init__(self, in_channels_seg, head_channels=32):
        super(TTNSeg_Dep2Inst, self).__init__()
        # Input 1 Semseg
        self.c1s = nn.Conv2d(in_channels_seg, 64, kernel_size=3, stride=1, padding=1)
        self.c2s = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1s = nn.BatchNorm2d(128)
        # Input 2 Depth
        self.c1d = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.c2d = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1d = nn.BatchNorm2d(128)

        self.c2m = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.c3m = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512) 
        self.c4m = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.en_depth = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.c5m = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.c6m = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.c7m = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.c8m = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(128)


        # Outputs
        self.head_channels = head_channels
        self.center_conv = nn.Conv2d(128, self.head_channels, 5, padding=2)
        self.center_pred = nn.Conv2d(self.head_channels, 1, 1)
        self.offset_conv = nn.Conv2d(128, self.head_channels, 5, padding=2)
        self.offset_pred = nn.Conv2d(self.head_channels, 2, 1)

    def forward(self, semseg, depth):
        s = self.c1s(semseg)
        s = self.c2s(s)
        s = F.leaky_relu(self.bn1s(s))

        d = self.c1d(depth)
        d = self.c2d(d)
        d = F.leaky_relu(self.bn1d(d))

        x = torch.cat([s, d], dim=1)

        x = self.c2m(x)
        x = self.c3m(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.c4m(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.en_depth(x)

        x = self.c5m(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.c6m(x)
        x = F.leaky_relu(self.bn5(x))
        x = self.c7m(x)
        x = F.leaky_relu(self.bn6(x))
        x = self.c8m(x)
        x = F.leaky_relu(self.bn7(x))

        # Instance Segmentation Output
        outputs = {}
        center = self.center_conv(x)
        center = self.center_pred(center)
        outputs['centers'] = center

        offset = self.offset_conv(x)
        offset = self.offset_pred(offset)
        outputs['offsets'] = offset
        
        return outputs
