'''
Function:
    定义一些常见的评估函数
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np


'''均方误差损失'''
class MSELoss():
    def __init__(self, reduction='mean'):
        assert reduction in ['sum', 'none', 'mean']
        self.reduction = reduction
        self.storage = {}
    '''定义前向传播'''
    def __call__(self, preditions, targets):
        self.storage.update({
            'preditions': preditions, 'targets': targets
        })
        loss = 0.5 * (preditions - targets) ** 2
        if self.reduction == 'none': return loss
        loss = getattr(loss, self.reduction)()
        return loss
    '''定义反向传播'''
    def backward(self):
        gradient = self.storage['preditions'] - self.storage['targets']
        if self.reduction == 'mean': gradient /= gradient.shape[0]
        return gradient


'''交叉熵损失'''
class CrossEntropy():
    def __init__(self, reduction='mean', eps=1e-12):
        assert reduction in ['sum', 'none', 'mean']
        self.reduction = reduction
        self.eps = eps
        self.storage = {}
    '''定义前向传播'''
    def __call__(self, preditions, targets):
        self.storage.update({
            'preditions': preditions, 'targets': targets
        })
        preditions = np.clip(preditions, self.eps, 1 - self.eps)
        loss = - targets * np.log(preditions) - (1 - targets) * np.log(1 - preditions)
        if self.reduction == 'none': return loss
        loss = getattr(loss, self.reduction)()
        return loss
    '''定义反向传播'''
    def backward(self):
        preditions, targets = self.storage['preditions'], self.storage['targets']
        preditions = np.clip(preditions, self.eps, 1 - self.eps)
        gradient = - (targets / preditions) + (1 - targets) / (1 - preditions)
        if self.reduction == 'mean': gradient /= gradient.shape[0]
        return gradient