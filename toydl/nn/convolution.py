'''
Function:
    实现卷积层
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import math
import numpy as np
from .module import Module
from ..utils import DataConverter, ImageConverter


'''2D卷积'''
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        # 格式为: (kernel_h, kernel_w)
        self.kernel_size = DataConverter.totuple(self.kernel_size)
        # 格式为: (stride_h, stride_w)
        self.stride = DataConverter.totuple(self.stride)
        # 格式为: (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        self.padding = DataConverter.toquaternion(self.padding)
        # 初始化权重
        thresh = 1 / math.sqrt(np.prod(self.kernel_size))
        self.weight = np.random.uniform(
            -thresh, thresh, (self.out_channels, self.in_channels, *self.kernel_size)
        )
        self.bias = np.zeros((self.out_channels, 1))
        # 初始化storage
        self.storage.update({
            'direction': {
                'weight': np.zeros(np.shape(self.weight)), 
                'bias': np.zeros(np.shape(self.bias))
            }
        })
    '''定义前向传播'''
    def forward(self, x):
        batch_size, num_channels, h, w = x.shape
        x_cols = ImageConverter.im2col(x, self.kernel_size, self.stride, self.padding)
        weight_cols = self.weight.reshape((self.out_channels, -1))
        self.storage.update({'x_cols': x_cols, 'weight_cols': weight_cols})
        feats = weight_cols.dot(x_cols) + self.bias
        feats = feats.reshape(self.outputsize(x)[1:] + (batch_size,)).transpose(3, 0, 1, 2)
        return feats
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        accumulated_gradient = accumulated_gradient.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        if self.training:
            # 计算梯度
            grad_w = accumulated_gradient.dot(self.storage['x_cols'].T).reshape(self.weight.shape)
            grad_b = np.sum(accumulated_gradient, axis=1, keepdims=True)
            # 根据梯度更新weight
            results = self.update(self.weight, grad_w, self.storage['direction']['weight'])
            self.weight, self.storage['direction']['weight'] = results['params'], results['direction']
            # 根据梯度更新bias
            results = self.update(self.bias, grad_b, self.storage['direction']['bias'])
            self.bias, self.storage['direction']['bias'] = results['params'], results['direction']
        accumulated_gradient = self.storage['weight_cols'].T.dot(accumulated_gradient)
        accumulated_gradient = ImageConverter.col2im(accumulated_gradient, self.storage['x'].shape, self.kernel_size, self.stride, self.padding)
        return accumulated_gradient
    '''返回参数数量'''
    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.bias.shape)
    '''特征输出大小'''
    def outputsize(self, inp):
        batch_size, num_channels, h, w = inp.shape
        output_size = (
            batch_size,
            self.out_channels,
            int((h + np.sum(self.padding[:2]) - self.kernel_size[0]) / self.stride[0] + 1),
            int((w + np.sum(self.padding[2:]) - self.kernel_size[1]) / self.stride[1] + 1),
        )
        return output_size