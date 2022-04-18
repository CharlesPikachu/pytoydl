'''
Function:
    随机数生成算法
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import numpy as np


# 随机种子
np.random.seed(689346065)
# 数据读取
fp = open('coffee.txt', 'r')
infos = []
for item in fp.readlines():
    increase = int(item.strip().split(' ')[-1])
    infos.append(increase)
infos = np.array(infos[::-1])[5:]
# 求平均数
mean = infos.mean()
# 求偏差
upper_diff = (infos[infos >= mean] - mean).mean()
low_diff = (mean - infos[infos < mean]).mean()
# 大于平均值和小于平均值的比例
proportion = (infos >= mean).sum() / len(infos)
# 随机预测一组数据
preds = []
for _ in range(len(infos)):
    if np.random.random() > proportion:
        preds.append(-np.random.randint(0, low_diff) + mean)
    else:
        preds.append(np.random.randint(0, upper_diff) + mean)
# 计算平均误差
errors = []
for idx in range(len(preds)):
    error = abs(infos[idx] - preds[idx]) / infos[idx]
    errors.append(error)
print(f'预测平均误差为: {sum(errors) / len(errors)}') 