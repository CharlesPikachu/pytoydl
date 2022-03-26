'''
Function:
    实现卷积层
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .module import Module


'''2d卷积'''
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__(
            in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=int(kernel_size),
            stride=int(stride), padding=int(padding), bias=int(bias)
        )
        assert (self.stride > 0) and (self.padding >=0) and (self.kernel_size >= 1) and (self.in_channels > 0) and (self.out_channels > 0)
