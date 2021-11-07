import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResnetBackbone(nn.Module):
    def __init__(self, layers=50, pretrained=True):
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(f'Only 50, 101, and 152 layer model are implemented for ResNet. Got {layers}')
    
        super(ResnetBackbone, self).__init__()
        pretrained_model = torchvision.models.__dict__[f'resnet{layers}'](pretrained=pretrained)
        
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']

        # Free the memory
        del pretrained_model

    def forward(self, x):
        metadata = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        metadata['stem'] = x
        x = self.maxpool(x)
        metadata['res2'] = self.layer1(x)
        metadata['res3'] = self.layer2(metadata['res2'])
        metadata['res4'] = self.layer3(metadata['res3'])
        
        return metadata['res4'], metadata


class ResnetBackbone5(nn.Module):
    def __init__(self, layers=50, pretrained=True):
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(f'Only 50, 101, and 152 layer model are implemented for ResNet. Got {layers}')

        super(ResnetBackbone5, self).__init__()
        pretrained_model = torchvision.models.__dict__[f'resnet{layers}'](pretrained=pretrained)
        self.layer4 = pretrained_model._modules['layer4']
        
        # Free the memory
        del pretrained_model

    def forward(self, x):
        x = self.layer4(x)
        
        return x

