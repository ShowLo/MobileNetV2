# -*- coding: UTF-8 -*-

import torch.nn as nn

class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV2, including Linear Bottlenecks and Inverted Residuals
    '''
    def __init__(self, in_channels_num, out_channels_num, stride, expansion_factor):
        '''
        根据输入通道数、输出通道数、卷积步长、升维系数初始化线性瓶颈单元及反向残差结构
        '''
        # 深度卷积输入/输出的通道数(Number of channels for Depthwise Convolution input/output)
        DW_channels_num = round(in_channels_num * expansion_factor)
        # 是否使用残差结构(Whether to use residual structure or not)
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if expansion_factor == 1:
            # 不升维，省去第一个深度卷积DW(Without expansion, the first depthwise convolution is omitted)
            self.conv = nn.Sequential(
                # 深度卷积(Depthwise Convolution)
                nn.Conv2d(in_channels=in_channels_num, out_channels=in_channels_num, kernel_size=3, stride=stride, padding=1, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=in_channels_num),
                nn.ReLU6(inplace=True),
                # 线性逐点卷积(Linear-PW)
                nn.Conv2d(in_channels=in_channels_num, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels_num)
            )
        else:
            # 升维(With expansion)
            self.conv = nn.Sequential(
                # 用于升维的逐点卷积(Pointwise Convolution for expansion)
                nn.Conv2d(in_channels=in_channels_num, out_channels=DW_channels_num, kernel_size=1, stride=1, padding=0, bias=False)
                nn.BatchNorm2d(num_features=DW_channels_num),
                nn.ReLU6(inplace=True),
                # 深度卷积(Depthwise Convolution)
                nn.Conv2d(in_channels=DW_channels_num, out_channels=DW_channels_num, kernel_size=3, stride=stride, padding=1, groups=DW_channels_num, bias=False),
                nn.BatchNorm2d(num_features=DW_channels_num),
                nn.ReLU6(inplace=True),
                # 线性逐点卷积(Linear-PW)
                nn.Conv2d(in_channels=DW_channels_num, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels_num)
            )

    def forward(self, x):
        '''
        前向传播
        '''
        if self.use_residual:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    '''
    
    '''
    def __init__(self, class_num=1000, input_size=224, width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        first_channel_num = 32
        last_channel_num = 1280
        bottleneck_setting = [
            # 升维系数(expansion factor), 输出通道数(number of output channels), 重复次数(repeat times), 卷积步长(stride)
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 输入层(input layer)
        first_channel_num = round(first_channel_num * width_multiplier)
        last_channel_num = round(last_channel_num * width_multiplier)
        # 按顺序加入各层网络(Join the layers of the network sequentially)
        self.features_net = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=first_channel_num, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_channel_num),
            nn.ReLU6(inplace=True)
        )
