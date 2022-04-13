import torch
from base import BaseModel
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from itertools import chain

# 参数初始化
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

'''
编码器
'''
# backbone
class Backbone(nn.Module):
    def __init__(self, in_channels=3, backbone='resnet50', pretrained=True):
        super(Backbone, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        s3, s4, d3, d4 = (1, 1, 2, 4)
        
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
            elif 'downsample.0' in n:
                m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x_mid, x_max):
        x_ = self.layer0(x_mid)
        x_ = self.layer1(x_)
        low_level_features = x_
        x = self.layer0(x_max)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return low_level_features,x

# ASSP
def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

class ASSP(nn.Module):
    def __init__(self, in_channels):
        super(ASSP, self).__init__()

        dilations = [1, 12, 24, 36]
        
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            nn.Dropout(0.5)
            )
        self.avg_pool_single = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.ReLU(inplace=True)
            nn.Dropout(0.5)
            )
        

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        if x.size(0) == 1:
            x5 = F.interpolate(self.avg_pool_single(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        else:
            x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

'''
解码器
'''
# Decoder
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.layer0 =  nn.Sequential(
            nn.Conv2d(low_level_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv1 = 
        self.layer1 =  nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Conv2d(128, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(512,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.adaptivepool2s =  nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc1 = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.output = nn.Sequential(
            self.fc0,
            self.fc1
        )
#         
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.layer0(low_level_features)
        x = torch.cat((low_level_features,x),1)
        x = self.layer1(x)
        x = self.adaptivepool2s(x)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        return self.output(x)

# 模型 RockSlice02
class RockSlice02(BaseModel):
    def __init__(self, num_classes, **_):
                
        super(RockSlice02, self).__init__()
        self.backbone = Backbone()
        low_level_channels = 256

        self.ASSP = ASSP(in_channels=2048)
        self.decoder = Decoder(low_level_channels, num_classes)


    def forward(self, x_mid,x_max):
        low_level_features,x = self.backbone(x_mid,x_max)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        return x

    # & Decoder / ASSP 使用可微学习率
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()