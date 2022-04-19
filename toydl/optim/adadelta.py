'''
Function:
    定义Adadelta优化器
Author:
    Zhenchao
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .baseoptimizer import BaseOptimizer


'''定义Adadelta优化器'''
class Adadelta(BaseOptimizer):
    def __init__(self, structure, criterion, learning_rate=1.0, rho=0.9, eps=1e-6):
        super(Adadelta, self).__init__(
            structure=structure, criterion=criterion, learning_rate=learning_rate, rho=rho, eps=eps
        )
        # 所有网络层都使用优化器的update函数
        self.applyupdate(self.structure.modules())
    '''更新函数'''
    def update(self, params, grads, direction, direction_2x):
        direction_2x = self.rho * direction_2x + (1 - self.rho) * np.power(grads, 2)
        learning_rate_adaptive = self.learning_rate * np.sqrt(direction + self.eps) / np.sqrt(direction_2x + self.eps)
        direction = self.rho * direction + (1 - self.rho) * np.power(learning_rate_adaptive * grads, 2)
        params = params - learning_rate_adaptive * grads
        return {
            'params': params, 'direction': direction, 'direction_2x': direction_2x,
        }