'''
Function:
    定义RMSprop优化器
Author:
    Zhenchao
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .baseoptimizer import BaseOptimizer


'''定义RMSprop优化器'''
class RMSprop(BaseOptimizer):
    def __init__(self, structure, criterion, learning_rate=0.01, rho=0.99, eps=1e-8):
        super(RMSprop, self).__init__(
            structure=structure, criterion=criterion, learning_rate=learning_rate, rho=rho, eps=eps,
        )
        # 所有网络层都使用优化器的update函数
        self.applyupdate(self.structure.modules())
    '''更新函数'''
    def update(self, params, grads, direction, direction_2x):
        direction_2x = self.rho * direction_2x + (1 - self.rho) * np.power(grads, 2)
        params = params - self.learning_rate * grads / np.sqrt(direction_2x + self.eps)
        return {
            'params': params, 'direction': direction, 'direction_2x': direction_2x
        }