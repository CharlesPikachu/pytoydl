'''
Function:
    定义激活函数
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .module import Module


'''softmax'''
class Softmax(Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__(dim=dim)
    '''定义前向传播'''
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = self.forward(self.storage['x'])
        gradient = gradient * (1 - gradient)
        return accumulated_gradient * gradient