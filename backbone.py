import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
import os
from resnet import resnet18, resnet50, resnet101

def process(style):
    eps=1e-5
    size = style.size()
    assert (len(size) == 4)
    N, C = size[:2]
    style_var = style.view(N, C, -1).var(dim=2) + eps
    style_std = style_var.sqrt().view(N, C)
    style_mean = style.view(N, C, -1).mean(dim=2).view(N, C)
    style_stat = torch.cat((style_mean, style_std), dim=1)
    return style_stat

class Backbone(nn.Module):
    def __init__(self, cfg):
        super(Backbone, self).__init__()

        if cfg.feat_extractor == 'resnet18':
            resnet = resnet18()
            resnet.load_state_dict(torch.load(os.path.join(cfg.model_root, 'resnet18-5c106cde.pth')),strict=True)
            #resnet = torchvision.models.resnet.resnet18(pretrained=True)
        elif cfg.feat_extractor == 'resnet50':
            resnet = resnet50()
            resnet.load_state_dict(torch.load(os.path.join(cfg.model_root, 'resnet50-19c8e357.pth')),strict=True)
            #resnet = torchvision.models.resnet.resnet50(pretrained=True)
        elif cfg.feat_extractor == 'resnet101':
            resnet = resnet101()
            resnet.load_state_dict(torch.load(os.path.join(cfg.model_root, 'resnet101-5d3b4d8f.pth')),strict=True)
            #resnet = torchvision.models.resnet.resnet101(pretrained=True)

        self.block0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4

    def forward(self, x, returned=[4]):
        blocks = [self.block0(x)]

        blocks.append(self.block1(blocks[-1]))
        blocks.append(self.block2(blocks[-1]))
        blocks.append(self.block3(blocks[-1]))
        blocks.append(self.block4(blocks[-1]))

        style_0, style_1, style_2, style_3 = process(blocks[0]), process(blocks[1]), process(blocks[2]), process(blocks[3])
        style_stat = torch.cat((style_0, style_1, style_2, style_3), dim=1)
        
        return blocks[4], style_stat, blocks[0], blocks[1], blocks[2], blocks[3]
        #out = [blocks[i] for i in returned]
        #out = F.adaptive_avg_pool2d(out[0],(1,1))
        #out = out.squeeze(-1).squeeze(-1)
        #return out

class new_model(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = torchvision.models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        # pdb.set_trace()
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x

class comb_resnet(nn.Module):
    def __init__(self):
        super(comb_resnet,self).__init__()
        self.l1 = new_model(output_layer = 'layer1').eval().cuda()
        self.l2 = new_model(output_layer = 'layer2').eval().cuda()
        self.l3 = new_model(output_layer = 'layer3').eval().cuda()
        self.l4 = new_model(output_layer = 'layer4').eval().cuda()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    
    def forward(self,img1):
        f1 = self.pool(self.l1(img1)) #.squeeze()
        f2 = self.pool(self.l2(img1)) #.squeeze()
        f3 = self.pool(self.l3(img1)) #.squeeze()
        f4 = self.pool(self.l4(img1)) #.squeeze()
        # pdb.set_trace()
        con = torch.cat((f1,f2,f3,f4),1)
        return con