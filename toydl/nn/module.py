'''
Function:
    网络模型基础类
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np


'''网络模型基础类'''
class Module():
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        # 默认为可训练状态
        self.training = True
        # 保存一些必要的数据用于反向传播
        self.storage = {}
    '''前向传播调用'''
    def __call__(self, x, **kwargs):
        self.storage.update({'x': x})
        x = self.forward(x, **kwargs)
        return x
    '''定义前向传播'''
    def forward(self):
        raise NotImplementedError('not to be implemented')
    '''定义反向传播'''
    def backward(self):
        raise NotImplementedError('not to be implemented')
    '''返回参数数量'''
    def parameters(self):
        return 0
    '''返回模型的名字'''
    def name(self):
        return self.__class__.__name__
    '''设置为训练状态'''
    def train(self, mode=True):
        def apply(module_dict, mode):
            for module in module_dict.values():
                if isinstance(module, dict):
                    apply(module, mode)
                else:
                    setattr(module, 'training', mode)
        module_dict = self.modules()
        apply(module_dict, mode)
    '''设置为测试状态'''
    def eval(self):
        self.train(mode=False)
    '''根据使用的优化器更新参数'''
    def update(self):
        raise NotImplementedError('not to be implemented')
    '''返回所有modules'''
    def modules(self):
        module_dict, attrs = {self.name(): self}, dir(self)
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, Module):
                module_dict[f'{self.name()}.{attr}'] = value.modules()
        return module_dict