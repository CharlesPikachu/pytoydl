'''
Function:
    实现Flatten层
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .module import Module


'''Flatten层'''
class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
    '''定义前向传播'''
    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        accumulated_gradient = accumulated_gradient.reshape(self.storage['x'].shape)
        return accumulated_gradient