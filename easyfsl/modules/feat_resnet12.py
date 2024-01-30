"""
This particular ResNet12 is simplified from the original implementation of FEAT (https://github.com/Sha-Lab/FEAT).
We provide it to allow the reproduction of the FEAT method and the use of the chekcpoints they made available.
It contains some design choices that differ from the usual ResNet12. Use this one or the other.
Just remember that it is important to use the same backbone for a fair comparison between methods.
"""

from torch import nn
from torchvision.models.resnet import conv3x3

import math

from tensorflow.keras import layers
import tensorflow as tf
from keras import backend


def squeeze_excitation_block(input_layer, out_dim, ratio=32, name=None):
	"""Squeeze and Extraction block
   Args:
      input_layer: input tensor
      out_dim: integer, output dimension for the model
      ratio: integer, reduction ratio for the number of neurons in the hidden layers
      name: string, block label
    Returns:
      Output A tensor for the squeeze and excitation block
    """

	#  Get the number of channels of the input characteristic graph
	in_channel = input_layer.shape[-1]

	#  Global average pooling [h,w,c]==>[None,c]
	squeeze = layers.GlobalAveragePooling2D(name=name + "_Squeeze_GlobalPooling")(input_layer)
	# [None,c]==>[1,1,c]
	squeeze = layers.Reshape(target_shape=(1, 1, in_channel))(squeeze)

	excitation = layers.Dense(units=out_dim / ratio, name=name + "_Excitation_FC_1")(squeeze)
	excitation = layers.Activation('relu', name=name + '_Excitation_FC_Relu_1')(excitation)

	excitation = layers.Dense(out_dim, name=name + "_Excitation_FC_2")(excitation)
	excitation = layers.Activation('sigmoid', name=name + '_Excitation_FC_Relu_2')(excitation)
	excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

	scale = layers.multiply([input_layer, excitation])

	return scale


def ECA_Net_block(input_layer, kernel_size=3, adaptive=False, name=None):
	"""ECA-Net: Efficient Channel Attention Block
    Args:
      input_layer: input tensor
      kernel_size: integer, default: 3, size of the kernel for the convolution
      adaptive: bool, default false , set kernel size depending on the number of input channels
      name: string, block label
    Returns:
      Output A tensor for the ECA-Net attention block
    """
	if adaptive:
		b = 1
		gamma = 2
		channels = input_layer.shape[-1]
		kernel_size = int(abs((math.log2(channels) + b / gamma)))
		if (kernel_size % 2) == 0:
			kernel_size = kernel_size + 1
		else:
			kernel_size = kernel_size

	squeeze = layers.GlobalAveragePooling2D(name=name + "_Squeeze_GlobalPooling")(input_layer)
	squeeze = tf.expand_dims(squeeze, axis=1)
	excitation = layers.Conv1D(filters=1,
							   kernel_size=kernel_size,
							   padding='same',
							   use_bias=False,
							   name=name + "_Excitation_Conv_1D")(squeeze)

	excitation = tf.expand_dims(tf.transpose(excitation, [0, 2, 1]), 3)
	excitation = tf.math.sigmoid(excitation)

	output = layers.multiply([input_layer, excitation])

	return output


def CBAM_block(input_layer, filter_num, reduction_ratio=32, kernel_size=7, name=None):
	"""CBAM: Convolutional Block Attention Module Block
    Args:
      input_layer: input tensor
      filter_num: integer, number of neurons in the hidden layers
      reduction_ratio: integer, default 32,reduction ratio for the number of neurons in the hidden layers
      kernel_size: integer, default 7, kernel size of the spatial convolution excitation convolution
      name: string, block label
    Returns:
      Output A tensor for the CBAM attention block
    """
	axis = -1

	# CHANNEL ATTENTION
	avg_pool = layers.GlobalAveragePooling2D(name=name + "_Chanel_AveragePooling")(input_layer)
	max_pool = layers.GlobalMaxPool2D(name=name + "_Chanel_MaxPooling")(input_layer)

	# Shared MLP
	dense1 = layers.Dense(filter_num // reduction_ratio, activation='relu', name=name + "_Chanel_FC_1")
	dense2 = layers.Dense(filter_num, name=name + "_Chanel_FC_2")

	avg_out = dense2(dense1(avg_pool))
	max_out = dense2(dense1(max_pool))

	channel = layers.add([avg_out, max_out])
	channel = layers.Activation('sigmoid', name=name + "_Chanel_Sigmoid")(channel)
	channel = layers.Reshape((1, 1, filter_num), name=name + "_Chanel_Reshape")(channel)

	channel_output = layers.multiply([input_layer, channel])

	# SPATIAL ATTENTION
	avg_pool2 = tf.reduce_mean(input_layer, axis=axis, keepdims=True)
	max_pool2 = tf.reduce_max(input_layer, axis=axis, keepdims=True)

	spatial = layers.concatenate([avg_pool2, max_pool2], axis=axis)

	# K = 7 achieves the highest accuracy
	spatial = layers.Conv2D(1, kernel_size=kernel_size, padding='same', name=name + "_Spatial_Conv2D")(spatial)
	spatial_out = layers.Activation('sigmoid', name=name + "_Spatial_Sigmoid")(spatial)

	CBAM_out = layers.multiply([channel_output, spatial_out])

	return CBAM_out


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
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(stride)
        
        self.downsample = downsample

    def forward(self, x):  # pylint: disable=invalid-name
        """
        Pass input through the block, including an activation and maxpooling at the end.
        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)


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
    ):
        self.inplanes = 3
        super().__init__()

        channels = [64, 160, 320]
        self.layer_dims = [
            channels[i] * block.expansion for i in range(3) for j in range(3)
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

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )



        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nn.functional.affine_grid(theta, x.size())
        x = nn.functional.grid_sample(x, grid)
        return x

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
        x = self.layer3(self.layer2(self.layer1(x)))
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



