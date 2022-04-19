'''
Function:
    定义SGD优化器
Author:
    Zhenchao
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .baseoptimizer import BaseOptimizer


'''定义SGD优化器'''
class SGD(BaseOptimizer):
    def __init__(self, structure, criterion, learning_rate=0.01, momentum=0):
        super(SGD, self).__init__(
            structure=structure, criterion=criterion, learning_rate=learning_rate, momentum=momentum
        )
        # 所有网络层都使用优化器的update函数
        self.applyupdate(self.structure.modules())
    '''更新函数'''
    def update(self, params, grads, direction, direction_2x):
        direction = self.momentum * direction + (1 - self.momentum) * grads
        params = params - self.learning_rate * direction
        return {
            'params': params, 'direction': direction, 'direction_2x': direction_2x
        }