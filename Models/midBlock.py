# MIT License
# 
# Copyright (c) 2018 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-04-16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class midBlock(nn.Module):

    def __init__(self, in_channels, output=20, kernel_size=(1,16) ,
                 stride = 1, groups = 1, dilation = 1, padding_type='SAME',pool_size=8, cuda=True):

        
        super(midBlock, self).__init__()
        self._cuda = cuda
        self.__dict__.update(locals())
        
        
        self.Conv2D = nn.Conv2d(in_channels,in_channels, 
                                kernel_size = self.kernel_size,groups=in_channels)
        self.pointwise = nn.Conv2d(self.in_channels,self.output, kernel_size=(1,1), groups=1)
        self.BN = nn.BatchNorm2d(self.output, momentum=0.01, affine=True )
        self.activation =nn.LeakyReLU()
        self.pool = nn.AvgPool2d(kernel_size = (1,pool_size), stride=(1,pool_size))
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
       
        identity = x
        
        x = self.Conv2D(x)
        x = self.pointwise(x)
        x = self.BN(x)
        
        x = self.activation(x)
        x = self.pool(x)
        x = self.drop(x)
        #print('Mid block: {}'.format(x.size()))
        return x


    @staticmethod
    def _get_padding(padding_type, kernel_size):
        #assert isinstance(kernel_size, int)
        assert padding_type in ['SAME', 'VALID'] 
        if padding_type == 'SAME':
            return (kernel_size - 1) // 2

        return 0
    @staticmethod
    def _calculate_output(H,padding,dilation, kernel_size, stride):
        
        numerator = (H + 2*padding-dilation * (kernel_size -1) - 1 )
        denominator = stride
        H_out = (numerator/denominator) + 1
        
        return H_out
    @staticmethod
    def _calculate_strided_padding(W, F, S):
         #  W= Input Size , F = filter size (kernel), S = stride,
        P = ((S-1)*W-S+F)//2
        
        return P
    @staticmethod
    def get_dilated_kernel(k,d):
        #kernel size, dilation
        new_k = k+(k-1)*(d-1)
        
        return new_k
