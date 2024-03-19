"""
This particular ResNet12 is simplified from the original implementation of FEAT (https://github.com/Sha-Lab/FEAT).
We provide it to allow the reproduction of the FEAT method and the use of the chekcpoints they made available.
It contains some design choices that differ from the usual ResNet12. Use this one or the other.
Just remember that it is important to use the same backbone for a fair comparison between methods.
"""

from torch import nn
from torchvision.models.resnet import conv3x3

'''Convolutional Block Attention Module (CBAM)
'''

import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.flatten import Flatten



class Channel_Attention(nn.Module):
    '''Channel Attention in CBAM.
    '''

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        '''Param init and architecture building.
        '''

        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''

        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.


class ChannelPool(nn.Module):
    '''Merge all the channels in a feature map into two separate channels where the first channel is produced by taking the max values from all channels, while the
       second one is produced by taking the mean from every channel.
    '''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_Attention(nn.Module):
    '''Spatial Attention in CBAM.
    '''

    def __init__(self, kernel_size=7):
        '''Spatial Attention Architecture.
        '''

        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = nn.Sigmoid()(x_output)
        return x * scaled


class CBAM(nn.Module):
    '''CBAM architecture.
    '''
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        '''Param init and arch build.
        '''
        super(CBAM, self).__init__()
        self.spatial = spatial

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out

class FEATBasicBlock(nn.Module):
    """
    BasicBlock for FEAT. Uses 3 convolutions instead of 2, a LeakyReLU instead of ReLU, and a MaxPool2d.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample

    def forward(self, x):  # pylint: disable=invalid-name
        """
        Pass input through the block, including an activation and maxpooling at the end.
        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        out = self.maxpool(out)

        return out


class FEATResNet12(nn.Module):
    """
    ResNet12 for FEAT. See feat_resnet12 doc for more details.
    """

    def __init__(
        self,
        block=FEATBasicBlock,
        use_cbam=True
    ):
        self.inplanes = 3
        super().__init__()
        
        self.use_cbam = use_cbam
        
        channels = [64, 160, 320, 640]
        self.layer_dims = [
            channels[i] * block.expansion for i in range(4) for j in range(4)
        ]

        self.layer1 = self._make_layer(
            block,
            64,
            stride=2,
        )
        self.layer2 = self._make_layer(
            block,
            160,
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
        )

        if self.use_cbam:
            self.cbam = CBAM(channel_in=channels[i]*self.expansion)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):  # pylint: disable=invalid-name
        """
        Iterate over the blocks and apply them sequentially.
        """
        x = self.layer4(self.cbam(self.layer3(self.layer2(self.layer1(x)))))
        return x.mean((-2, -1))


def feat_resnet12(**kwargs):
    """
    Build a ResNet12 model as used in the FEAT paper, following the implementation of
    https://github.com/Sha-Lab/FEAT.
    This ResNet network also follows the practice of the following papers:
    TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
    A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

    There are 4 main differences with the other ResNet models used in EasyFSL:
        - There is no first convolutional layer (3x3, 64) before the first block.
        - The stride of the first block is 2 instead of 1.
        - The BasicBlock uses 3 convolutional layers, instead of 2 in the standard torch implementation.
        - We don't initialize the last fully connected layer, since we never use it.

    Note that we removed the Dropout logic from the original implementation, as it is not part of the paper.

    Args:
        **kwargs: Additional arguments to pass to the FEATResNet12 class.

    Returns:
        The standard ResNet12 from FEAT model.
    """
    return FEATResNet12(FEATBasicBlock, **kwargs)
