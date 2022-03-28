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


'''Softmax'''
class Softmax(Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__(dim=dim)
    '''定义前向传播'''
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        p = self.forward(self.storage['x'])
        gradient = p * (1 - p)
        return accumulated_gradient * gradient


'''Sigmoid'''
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    '''定义前向传播'''
    def forward(self, x):
        x = 1 / (1 + np.exp(-x))
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        p = self.forward(self.storage['x'])
        gradient = p * (1 - p)
        return accumulated_gradient * gradient


'''Tanh'''
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
    '''定义前向传播'''
    def forward(self, x):
        x = 2 / (1 + np.exp(-2 * x)) - 1
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        p = self.forward(self.storage['x'])
        gradient = 1 - np.power(p, 2)
        return accumulated_gradient * gradient


'''ReLU'''
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
    '''定义前向传播'''
    def forward(self, x):
        x = np.where(x >= 0, x, 0)
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = np.where(self.storage['x'] >= 0, 1, 0)
        return accumulated_gradient * gradient


'''LeakyReLU'''
class LeakyReLU(Module):
    def __init__(self, alpha=0.2):
        super(LeakyReLU, self).__init__(alpha=alpha)
    '''定义前向传播'''
    def forward(self, x):
        x = np.where(x >= 0, x, self.alpha * x)
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = np.where(self.storage['x'] >= 0, 1, self.alpha)
        return accumulated_gradient * gradient


'''ELU'''
class ELU(Module):
    def __init__(self, alpha=0.1):
        super(ELU, self).__init__(alpha=alpha)
    '''定义前向传播'''
    def forward(self, x):
        x = np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = np.where(self.storage['x'] >= 0.0, 1, self.alpha * np.exp(self.storage['x']))
        return accumulated_gradient * gradient


'''SELU'''
class SELU(Module):
    def __init__(self):
        super(SELU, self).__init__(
            alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946
        )
    '''定义前向传播'''
    def forward(self, x):
        x = self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = self.scale * np.where(self.storage['x'] >= 0.0, 1, self.alpha * np.exp(self.storage['x']))
        return accumulated_gradient * gradient


'''Softplus'''
class Softplus(Module):
    def __init__(self):
        super(Softplus, self).__init__()
    '''定义前向传播'''
    def forward(self, x):
        x = np.log(1 + np.exp(x))
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = 1 / (1 + np.exp(-x))
        return accumulated_gradient * gradient