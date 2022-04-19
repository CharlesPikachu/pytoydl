'''
Function:
    定义Adagrad优化器
Author:
    Zhenchao
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .baseoptimizer import BaseOptimizer


'''定义Adagrad优化器'''
class Adagrad(BaseOptimizer):
    def __init__(self, structure, criterion, learning_rate=0.01, eps=1e-8):
        super(Adagrad, self).__init__(
            structure=structure, criterion=criterion, learning_rate=learning_rate, eps=eps
        )
        # 所有网络层都使用优化器的update函数
        self.applyupdate(self.structure.modules())
    '''更新函数'''
    def update(self, params, grads, direction, direction_2x):
        direction = grads
        direction_2x += np.power(grads, 2)
        params = params - self.learning_rate * direction / np.sqrt(direction_2x + self.eps)
        return {
            'params': params, 'direction': direction, 'direction_2x': direction_2x,
        }