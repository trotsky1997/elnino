

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import math
import torch_optimizer as optim



# In[28]:


import random
import os



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# In[43]:


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# In[44]:


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        x = self.ca(x) * x
        x = self.sa(x) * x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,input_chan=3,output_shape=(1,1),output_chan = 24):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv3x3(input_chan, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()

#         self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, input_chan*2, layers[0])
        self.layer2 = self._make_layer(block, input_chan*3, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, input_chan*4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, output_chan , layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(self.inplanes)
        self.sa1 = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d(output_shape)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                nn.AvgPool2d(kernel_size=(stride,stride)),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ca(x) * x
        x = self.sa(x) * x


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ca1(x) * x
        x = self.sa1(x) * x
        


#         x = self.avgpool(x)
#         x = x.view(x.shape[0],x.shape[1],-1)


        return x


# In[45]:


m = ResNet(BasicBlock,[2,2,2,2],input_chan=12)
a = torch.randn(32,12,24,72)


# In[46]:


m(a).shape


# In[47]:


def get_mma_loss(weight):
    '''
    MMA regularization in PyTorch
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    # for convolutional layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())

    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))

    # maxmize the minimum angle
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss


# In[48]:


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 3):
                pe[pos, i] =                     math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] =                     math.sin(pos / (10000 ** ((2 * (i + 1)) / d_model))+math.pi/3)
                pe[pos,i+2] =                      math.sin(pos / (10000 ** ((2 * (i + 1)) / d_model))+math.pi/3*2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


# In[49]:


class p2e(nn.Module):
    def __init__(self):
        super().__init__()
        k=[1,3,3,3,3]
        chans=self.chans= [12,32,64,96,128,24]
        self.planes = 9
        n_lstm_units=128
#         self.convs = nn.ModuleList([nn.Sequential(*[nn.Sequential(*([nn.Conv2d(in_channels=chans[i], out_channels=chans[i+1], kernel_size=[k[i],k[i]]),nn.BatchNorm2d(chans[i+1]),nn.ReLU()]+ ([nn.AvgPool2d((2,2)),] if i >=3 else []) )) for i in range(len(k))]) for i in range(self.planes)])
        self.convs = nn.ModuleList([ResNet(BasicBlock,[2,2,2,2],input_chan=12,output_chan = 24,output_shape=(4,12)) for i in range(self.planes)])
        self.bn3 = nn.BatchNorm1d(chans[-1])
#         self.conv3 = nn.Conv1d(24,24,kernel_size=3)
        self.lstm = nn.GRU(243, n_lstm_units, 2, bidirectional=True)
        self.o = nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
#         self.ct = nn.Linear(24,24)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        batch = x.shape[0]
        xs = [x[:,i] for i in range(4)]
        xs.append(torch.arctan(xs[0]/xs[1]))
        xs.append(torch.arctan(xs[3]/xs[2]))
        xs.append((xs[3]**2+xs[2]**2)**0.5)
        xs.append(torch.arctan(xs[6]/xs[0]))
        xs.append(torch.arctan(xs[6]/xs[1]))
        for i in range(self.planes):
            xs[i][xs[i]!=xs[i]] = 0
            xs[i] =  self.convs[i](xs[i])
        x = torch.stack(xs, dim=1).transpose(1,2)
        x = x.reshape(x.shape[0],self.chans[-1],-1)
#         x = self.conv3(x)
        x = self.bn3(x)
        attn = F.softmax(x.bmm(x.transpose(-1,-2))/math.sqrt(x.shape[-1]),dim=-1).bmm(x)
        x = x +attn

        x = self.dropout(x)
        x = x .transpose(0,1)
        x, h = self.lstm(x)
        x = self.o(x).squeeze(-1).transpose(0,1)
#         x = self.ct(x)
        return x


