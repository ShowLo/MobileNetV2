# -*- coding: UTF-8 -*-

'''
MobileNetV2
Ref: https://github.com/tonylins/pytorch-mobilenet-v2
'''

import torch.nn as nn

def _ensure_divisible(number, divisor, min_value=None):
    '''
    确保number可以被divisor整除
    Ensure that 'number' can be 'divisor' divisible
    Ref:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num

class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV2, including Linear Bottlenecks and Inverted Residuals
    '''
    def __init__(self, in_channels_num, out_channels_num, stride, expansion_factor):
        '''
        根据输入通道数、输出通道数、卷积步长、升维系数初始化线性瓶颈单元及反向残差结构
        '''
        super(Bottleneck, self).__init__()
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
                nn.BatchNorm2d(num_features=out_channels_num)
            )
        else:
            # 升维(With expansion)
            self.conv = nn.Sequential(
                # 用于升维的逐点卷积(Pointwise Convolution for expansion)
                nn.Conv2d(in_channels=in_channels_num, out_channels=DW_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=DW_channels_num),
                nn.ReLU6(inplace=True),
                # 深度卷积(Depthwise Convolution)
                nn.Conv2d(in_channels=DW_channels_num, out_channels=DW_channels_num, kernel_size=3, stride=stride, padding=1, groups=DW_channels_num, bias=False),
                nn.BatchNorm2d(num_features=DW_channels_num),
                nn.ReLU6(inplace=True),
                # 线性逐点卷积(Linear-PW)
                nn.Conv2d(in_channels=DW_channels_num, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels_num)
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
    def __init__(self, classes_num=1000, input_size=224, width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        first_channel_num = 32
        last_channel_num = 1280
        divisor = 8
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

        ########################################################################################################################
        # 特征提取部分(feature extraction part)
        # 输入层(input layer)
        input_channel_num = _ensure_divisible(first_channel_num * width_multiplier, divisor)
        last_channel_num = _ensure_divisible(last_channel_num * width_multiplier, divisor) if width_multiplier > 1 else last_channel_num
        self.network = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channel_num, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channel_num),
            nn.ReLU6(inplace=True)
        )
        self.network.append(first_layer)
        # 多个bottleneck结构叠加(Overlay of multiple bottleneck structures)
        # 按顺序加入各层网络(Join the layers of the network sequentially)
        for t, c, n, s in bottleneck_setting:
            output_channel_num = _ensure_divisible(c * width_multiplier, divisor)
            for i in range(n):
                if i == 0:
                    # 每一个bottleneck的第一层做步长>=1的卷积操作(The first layer of each bottleneck performs the convolution with stride>=1)
                    self.network.append(Bottleneck(in_channels_num=input_channel_num, out_channels_num=output_channel_num, stride=s, expansion_factor=t))
                    input_channel_num = output_channel_num
                else:
                    # 每一个bottleneck的之后每一层的卷积操作步长均为1(The later layers of the bottleneck perform the convolution with stride=1)
                    self.network.append(Bottleneck(in_channels_num=input_channel_num, out_channels_num=output_channel_num, stride=1, expansion_factor=t))
        # 最后几层(the last several layers)
        self.network.append(
            nn.Sequential(
                nn.Conv2d(in_channels=input_channel_num, out_channels=last_channel_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_channel_num),
                nn.ReLU6(inplace=True)
            )
        )
        self.network.append(
            nn.AvgPool2d(kernel_size=input_size//32, stride=1)
        )
        self.network = nn.Sequential(*self.network)

        ########################################################################################################################
        # 分类部分(Classification part)
        self.classifier = nn.Linear(last_channel_num, classes_num)

        ########################################################################################################################
        # 权重初始化(Initialize the weights)
        self._initialize_weights()

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        '''
        初始化权重
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)