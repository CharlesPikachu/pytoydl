'''
Function:
    定义Adam优化器
Author:
    Zhenchao
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .baseoptimizer import BaseOptimizer


'''定义Adam优化器'''
class Adam(BaseOptimizer):
    def __init__(self, structure, criterion, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8):
        super(Adam, self).__init__(
            structure=structure, criterion=criterion, learning_rate=learning_rate, betas=betas, eps=eps,
        )
        # 所有网络层都使用优化器的update函数
        self.applyupdate(self.structure.modules())
    '''更新函数'''
    def update(self, params, grads, direction, direction_2x):
        direction = self.betas[0] * direction + (1 - self.betas[0]) * grads
        direction_2x = self.betas[1] * direction_2x + (1 - self.betas[1]) * np.power(grads, 2)
        m, v = direction / (1 - direction), direction_2x / (1 - direction_2x)
        params = params - self.learning_rate * m / (np.sqrt(v) + self.eps)
        return {
            'params': params, 'direction': direction, 'direction_2x': direction_2x
        }