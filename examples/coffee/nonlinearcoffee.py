'''
Function:
    随机数生成算法
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import math
import toydl
import numpy as np
import toydl.nn as nn


# 随机种子
np.random.seed(689346065)
# 数据读取
fp = open('coffee.txt', 'r')
infos = []
for item in fp.readlines():
    increase = int(item.strip().split(' ')[-1])
    infos.append(increase)
infos = np.array(infos[::-1])[5:]
# 数据预处理
mean = infos.mean()
upper_diff = (infos[infos >= mean] - mean).mean()
low_diff = (mean - infos[infos < mean]).mean()
proportion = (infos >= mean).sum() / len(infos)
inputs = []
for _ in range(len(infos)):
    if np.random.random() > proportion:
        inputs.append(-np.random.randint(0, low_diff) + mean)
    else:
        inputs.append(np.random.randint(0, upper_diff) + mean)
inputs = np.array(inputs)
inputs = (inputs - inputs.mean()) / (inputs.max() - inputs.min())
labels = (infos - infos.mean()) / (infos.max() - infos.min())
# 模型拟合
model = nn.Sequential()
model.addmodule('fc1', nn.Linear(1, 128))
model.addmodule('leakyrelu1', nn.LeakyReLU())
model.addmodule('fc2', nn.Linear(128, 128))
model.addmodule('leakyrelu2', nn.LeakyReLU())
model.addmodule('fc3', nn.Linear(128, 1))
criterion = nn.MSELoss()
optimzer = toydl.optim.SGD(model, criterion=criterion, learning_rate=0.01, momentum=0.9)
losses, batch_size = [], 2
for epoch in range(360):
    for idx in range(math.ceil(inputs.shape[0] / batch_size)):
        x_b = np.array(inputs[idx * batch_size: (idx + 1) * batch_size]).reshape(batch_size, 1)
        t_b = np.array(labels[idx * batch_size: (idx + 1) * batch_size]).reshape(batch_size, 1)
        output = model(x_b)
        loss = criterion(output, t_b)
        optimzer.step()
        losses.append(loss)
        if len(losses) == 5:
            print(f'Epoch: {epoch+1}/10, Batch: {idx+1}/{math.ceil(inputs.shape[0])}, Loss: {sum(losses) / len(losses)}')
            losses = []
    predictions = model(inputs.reshape(-1, 1))
    predictions = predictions * (infos.max() - infos.min()) + infos.mean()
    errors = []
    for idx in range(len(inputs)):
        errors.append(abs(predictions[idx] - infos[idx]) / infos[idx])
    print(f'拟合至{epoch+1}个Epoch的预测平均误差为: {sum(errors) / len(errors)}')