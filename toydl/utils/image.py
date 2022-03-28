'''
Function:
    定义一些和图像相关的工具函数
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np


'''图像格式转换工具'''
class ImageConverter():
    '''图像转column'''
    @staticmethod
    def im2col(image, kernel_size, stride, padding):
        padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        image_padded = np.pad(image, ((0, 0), (0, 0), padding[0], padding[1]), mode='constant')
        k, i, j = ImageConverter.im2colindices(image.shape, kernel_size, stride, padding)
        cols = image_padded[:, k, i, j]
        cols = cols.transpose(1, 2, 0).reshape(kernel_size[0] * kernel_size[1] * image.shape[1], -1)
        return cols
    '''column转图像'''
    @staticmethod
    def col2im(cols, image_size, kernel_size, stride, padding):
        padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        batch_size, num_channels, h, w = image_size
        image_padded = np.zeros((batch_size, num_channels, h + np.sum(padding[0]), w + np.sum(padding[1])))
        k, i, j = ImageConverter.im2colindices(image_size, kernel_size, stride, padding)
        cols = cols.reshape(num_channels * np.prod(kernel_size), -1, batch_size)
        cols = cols.transpose(2, 0, 1)
        np.add.at(image_padded, (slice(None), k, i, j), cols)
        pad_h, pad_w = padding
        return image_padded[:, :, pad_h[0]: h+pad_h[0], pad_w[0]: w+pad_w[0]]
    '''获得im2col的索引, Reference: CS231n Stanford'''
    @staticmethod
    def im2colindices(image_size, kernel_size, stride, padding):
        batch_size, num_channels, h, w = image_size
        out_h = int((h + np.sum(padding[0]) - kernel_size[0]) / stride[0] + 1)
        out_w = int((w + np.sum(padding[1]) - kernel_size[1]) / stride[1] + 1)
        # i0
        i0 = np.repeat(np.arange(kernel_size[0]), kernel_size[1])
        i0 = np.tile(i0, num_channels)
        # i1
        i1 = stride[0] * np.repeat(np.arange(out_h), out_w)
        # j0
        j0 = np.tile(np.arange(kernel_size[1]), kernel_size[0] * num_channels)
        # j1
        j1 = stride[1] * np.tile(np.arange(out_w), out_h)
        # i, j, k
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(num_channels), kernel_size[0] * kernel_size[1]).reshape(-1, 1)
        return (k, i, j)