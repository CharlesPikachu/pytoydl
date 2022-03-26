'''
Function:
    定义Sequential模型
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from .module import Module


'''定义Sequential模型'''
class Sequential(Module):
    def __init__(self, module_list=[], **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.module_dict = {}
        for idx, module in enumerate(module_list):
            self.module_dict.update({str(idx): module})
    '''添加模型'''
    def addmodule(self, name, module):
        self.module_dict.update({name: module})
    '''前向传播'''
    def forward(self, x):
        for module in self.module_dict.values():
            x = module(x)
        return x
    '''反向传播'''
    def backward(self, accumulated_gradient):
        for module in self.module_dict.values():
            accumulated_gradient = module.backward(accumulated_gradient)
        return accumulated_gradient
    '''返回参数数量'''
    def parameters(self):
        num_paramters = 0
        for module in self.module_dict.values(): 
            num_paramters += module.parameters()
        return num_paramters
    '''返回所有modules'''
    def modules(self):
        return self.module_dict