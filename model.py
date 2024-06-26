import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ChallengeDataset
import pandas as pd


class Flatten(nn.Module):
    """
    Custom module to flatten the input tensor.
    This module flattens the input tensor to a 2D tensor while preserving the batch dimension. 
    """
    def __init__(self):
        """
        Initializes a Flatten Module
        """
        super(Flatten, self).__init__()
        self.batch_dim = None

    def forward(self, input_tensor):
        """
        Forward pass of the Flatten module.
        
        Parameters:
            input_tensor (torch.Tensor): Input tensor to be flattened.

        Returns:
            torch.Tensor: Flattened tensor.
        """
        self.batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(self.batch_dim, -1)


class ResBlock(nn.Module):
    """
    Residual block module for ResNet architecture. 
    """
    def __init__(self, in_channels, out_channels, stride_shape=1):
        """
        Initializes a ResBlock module
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride_shape (int, optional): Stride size for convolution operation. Default is 1.   
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_shape, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = True
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape)
        if in_channels == out_channels and stride_shape == 1:
            self.residual_conv = False
        else:
            self.residual_conv = True

        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU()
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu_1, self.conv2, self.batch_norm2)
        self.residual = None

    def forward(self, input_tensor):
        """
        Forward pass of the Residual block module.

        Parameters:
            input_tensor (torch.Tensor): Input tensor to pass through the residual block.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        self.residual = input_tensor
        output_tensor = self.seq(input_tensor)
        if self.residual_conv:
            self.residual = self.conv1X1(self.residual)
        # Now normalize the residual
        self.residual = self.batch_norm3(self.residual)
        output_tensor += self.residual
        output_tensor = self.relu_3(output_tensor)
        return output_tensor


class ResNet(nn.Module):
    """
    ResNet model architecture consisting of the residual blocks, the average pooling layer and the final classification layer.
    """
    def __init__(self):
        """
        Initializes a ResNet model.
        """
        super(ResNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.seq2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=128, stride_shape=2),
            ResBlock(in_channels=128, out_channels=256, stride_shape=2),
            nn.Dropout(p=0.5),
            ResBlock(in_channels=256, out_channels=512, stride_shape=2)
        )

        self.seq3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=10),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        """
        Forward pass for the ResNet model.

        Parameters:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the ResNet model.
        """
        output_tensor = self.seq1(input_tensor)
        output_tensor = self.seq2(output_tensor)
        output_tensor = self.seq3(output_tensor)
        return output_tensor


