'''
Function:
    定义基类优化器
Author:
    Zhenchao
微信公众号:
    Charles的皮卡丘
'''
import numpy as np


'''定义基类优化器'''
class BaseOptimizer():
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
    '''所有网络层都使用优化器的update函数'''
    def applyupdate(self, module_dict):
        for module in module_dict.values():
            if isinstance(module, dict):
                self.applyupdate(module)
            else:
                setattr(module, 'update', self.update)
    '''梯度更新函数'''
    def update(self, params, grads, direction):
        raise NotImplementedError('not to be implemented')
    '''参数更新'''
    def step(self):
        self.structure.backward(self.criterion.backward())