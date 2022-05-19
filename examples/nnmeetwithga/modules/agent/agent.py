'''
Function:
    定义AIAgent
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import copy
import random
import numpy as np
import toydl.nn as nn
from ..sprites import Dinosaur


'''定义AIAgent'''
class AIAgent():
    def __init__(self, images, sounds, population_size=100, **kwargs):
        self.images = images
        self.sounds = sounds
        self.population_size = population_size
        self.dinos = {}
        for idx in range(self.population_size):
            self.dinos[idx] = {
                'sprite': Dinosaur(images['dino']),
                'network': nn.Sequential([nn.Linear(5, 16, bias=False), nn.Tanh(), nn.Linear(16, 2, bias=False), nn.Tanh()]),
                'fitness': 0,
            }
    '''更新所有小鸟'''
    def update(self):
        for dino in self.dinos.values():
            if dino['sprite'].is_dead: continue
            dino['sprite'].update()
    '''画到屏幕上'''
    def draw(self, screen):
        for dino in self.dinos.values():
            if dino['sprite'].is_dead: continue
            dino['sprite'].draw(screen)
    '''决定小鸟的行动'''
    def decide(self, x):
        x = np.array(x).reshape(1, -1)
        for key, value in self.dinos.items():
            action = value['network'](x)[0]
            if value['sprite'].is_dead: continue
            if action[0] >= 0.55: value['sprite'].jump(self.sounds)
            elif action[1] >= 0.55: value['sprite'].duck()
            else: value['sprite'].unduck()
            value['fitness'] = value['sprite'].score
    '''生成下一代'''
    def nextgeneration(self):
        # 保留top2%
        dinos = list(self.dinos.values())
        dinos.sort(key=lambda x: x['fitness'], reverse=True)
        dinos = dinos[:int(self.population_size * 0.02)]
        # 交叉变异
        generated_dinos = self.crossover(dinos)
        for dino in generated_dinos:
            dinos.append(self.mutate(dino))
            dinos[-1]['fitness'] = 0
        size = self.population_size - len(dinos)
        if size > 0:
            for idx in range(size):
                dinos.append(self.mutate(random.choice(dinos)))
                dinos[-1]['fitness'] = 0
        for idx in range(self.population_size):
            self.dinos[idx] = dinos[idx]
    '''交叉'''
    def crossover(self, dinos, num_crossover_times=2):
        def crossoverdinos(dino_1, dino_2):
            network_1, network_2 = dino_1['network'], dino_2['network']
            weight_1_1, weight_1_2 = network_1.module_dict['0'].weight, network_1.module_dict['2'].weight
            weight_2_1, weight_2_2 = network_2.module_dict['0'].weight, network_2.module_dict['2'].weight
            # 变异第一层
            num_crossovers = int(len(weight_1_1) * random.uniform(0, 1))
            for idx in range(num_crossovers):
                weight_1_1[idx], weight_2_1[idx] = weight_2_1[idx].copy(), weight_1_1[idx].copy()
            # 变异第二层
            num_crossovers = int(len(weight_1_2) * random.uniform(0, 1))
            for idx in range(num_crossovers):
                weight_1_2[idx], weight_2_2[idx] = weight_2_2[idx].copy(), weight_1_2[idx].copy()
        generated_dinos = []
        size = min(int(self.population_size * 0.02) * int(self.population_size * 0.02), self.population_size)
        for idx in range(size):
            dino_1 = self.deepcopy(random.choice(dinos))
            dino_2 = self.deepcopy(random.choice(dinos))
            for _ in range(num_crossover_times):
                crossoverdinos(dino_1, dino_2)
                generated_dinos.append(dino_1)
        return generated_dinos
    '''变异'''
    def mutate(self, dino, prob=0.5):
        dino = self.deepcopy(dino)
        if random.uniform(0, 1) < prob:
            dino['network'].module_dict['0'].weight = dino['network'].module_dict['0'].weight * random.uniform(0.5, 1.5)
        if random.uniform(0, 1) < prob:
            dino['network'].module_dict['2'].weight = dino['network'].module_dict['2'].weight * random.uniform(0.5, 1.5)
        return dino
    '''深拷贝'''
    def deepcopy(self, dino):
        dino_copied = {
            'sprite': dino['sprite'],
            'network': copy.deepcopy(dino['network']),
            'fitness': dino['fitness'],
        }
        return dino_copied