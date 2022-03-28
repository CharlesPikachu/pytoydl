'''
Function:
    定义一些和数据相关的工具函数
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np


'''数据格式转换工具'''
class DataConverter():
    '''转元组类型'''
    @staticmethod
    def totuple(inp):
        if isinstance(inp, tuple) or isinstance(inp, list):
            return tuple(inp)
        if isinstance(inp, float): 
            inp = int(inp)
        if isinstance(inp, int): 
            inp = (inp, inp)
        else: 
            raise TypeError(f'data type {type(inp)} can not be converted to tuple')
        return inp
    '''转四元数'''
    @staticmethod
    def toquaternion(inp):
        if isinstance(inp, tuple) or isinstance(inp, list):
            assert len(inp) == 4
            return inp
        if isinstance(inp, float): 
            inp = int(inp)
        if isinstance(inp, int): 
            inp = (inp, inp, inp, inp)
        else:
            raise TypeError(f'data type {type(inp)} can not be converted to quaternion')
        return inp