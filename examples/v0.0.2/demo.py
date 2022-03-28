'''
Function:
    "从零开始实现一个深度学习框架 | 激活函数，损失函数与卷积层"完整示例代码
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import math
import toydl
import random
import numpy as np
import toydl.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets


# 定义卷积网络
cnn = nn.Sequential()
cnn.addmodule('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
cnn.addmodule('relu1', nn.ReLU())
cnn.addmodule('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
cnn.addmodule('relu2', nn.ReLU())
cnn.addmodule('conv3', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
cnn.addmodule('flatten', nn.Flatten())
cnn.addmodule('fc', nn.Linear(32 * 8 * 8, 10))
cnn.addmodule('softmax', nn.Softmax())
# 定义损失函数
criterion = nn.CrossEntropy()
# 定义优化器
optimzer = toydl.optim.SGD(cnn, criterion=criterion, learning_rate=0.001, momentum=0.9)
# 导入数据集
# --手写数字数据集
data = datasets.load_digits()
inputs, targets = data.data.reshape(-1, 1, 8, 8), data.target
'''
# 这两行代码可以让你看到数据集里的图片到底长啥样
import cv2
cv2.imwrite('1.jpg', inputs[0].reshape(8, 8) / 16 * 255)
'''
# --targets转成one-hot格式
num_classes = np.amax(targets) + 1
one_hot = np.zeros((targets.shape[0], num_classes))
one_hot[np.arange(targets.shape[0]), targets] = 1
targets = one_hot
# --随机打乱划分训练集和验证集
inds = list(range(inputs.shape[0]))
random.shuffle(inds)
trainset = inputs[inds[:int(inputs.shape[0] * 0.9)]], targets[inds[:int(inputs.shape[0] * 0.9)]]
valset = inputs[inds[int(inputs.shape[0] * 0.9):]], targets[inds[int(inputs.shape[0] * 0.9):]]
# 开始训练
losses_log, batch_size, losses_batch, vis_infos = [], 16, [], []
for epoch in range(60):
    for idx in range(math.ceil(trainset[0].shape[0] / batch_size)):
        x_b = trainset[0][idx * batch_size: (idx + 1) * batch_size] / 16
        t_b = trainset[1][idx * batch_size: (idx + 1) * batch_size]
        output = cnn(x_b)
        loss = criterion(output, t_b)
        optimzer.step()
        losses_log.append(loss)
        losses_batch.append(loss)
        if len(losses_log) == 20:
            print(f'Epoch: {epoch+1}/10, Batch: {idx+1}/{math.ceil(trainset[0].shape[0] / 32)}, Loss: {sum(losses_log) / len(losses_log)}')
            losses_log = []
    # 每个epoch结束之后测试一下模型准确性
    predictions = cnn(valset[0]).argmax(1)
    acc = np.equal(predictions, valset[1].argmax(1)).sum() / predictions.shape[0]
    print(f'Accuracy of Epoch {epoch+1} is {acc}')
    vis_infos.append([acc, sum(losses_batch) / len(losses_batch)])
    losses_batch = []
# 画下loss曲线和acc曲线
plt.plot(range(60), [i[0] for i in vis_infos], marker='o', label='Accuracy')
plt.plot(range(60), [i[1] for i in vis_infos], marker='*', label='Loss')
plt.legend()
plt.show()