'''
Function:
    "从零开始实现一个深度学习框架 | 当神经网络遇上遗传算法"完整示例代码
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import sys
import time
import random
import pygame
from modules import Ptera, Dinosaur, Cactus, Ptera, Ground, Cloud, Scoreboard, AIAgent


'''游戏配置信息'''
class Config():
    # 根目录
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    # FPS
    FPS = 60
    # 标题
    TITLE = 'T-Rex Rush —— Charles的皮卡丘'
    # 背景颜色
    BACKGROUND_COLOR = (235, 235, 235)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    # 屏幕大小
    SCREENSIZE = (600, 150)
    # 游戏图片路径
    IMAGE_PATHS_DICT = {
        'cacti': [os.path.join(rootdir, 'resources/images/cacti-big.png'), os.path.join(rootdir, 'resources/images/cacti-small.png')],
        'cloud': os.path.join(rootdir, 'resources/images/cloud.png'),
        'dino': [os.path.join(rootdir, 'resources/images/dino.png'), os.path.join(rootdir, 'resources/images/dino_ducking.png')],
        'gameover': os.path.join(rootdir, 'resources/images/gameover.png'),
        'ground': os.path.join(rootdir, 'resources/images/ground.png'),
        'numbers': os.path.join(rootdir, 'resources/images/numbers.png'),
        'ptera': os.path.join(rootdir, 'resources/images/ptera.png'),
        'replay': os.path.join(rootdir, 'resources/images/replay.png')
    }
    # 游戏声音路径
    SOUND_PATHS_DICT = {
        'die': os.path.join(rootdir, 'resources/audios/die.wav'),
        'jump': os.path.join(rootdir, 'resources/audios/jump.wav'),
        'point': os.path.join(rootdir, 'resources/audios/point.wav')
    }


'''仿谷歌浏览器小恐龙游戏'''
class TRexRushGame():
    def __init__(self, **kwargs):
        self.cfg = Config()
    '''运行游戏'''
    def run(self):
        # 游戏初始化
        cfg = self.cfg
        pygame.init()
        screen = pygame.display.set_mode(cfg.SCREENSIZE)
        pygame.display.set_caption(cfg.TITLE)
        # 导入所有声音
        sounds = {}
        for key, value in cfg.SOUND_PATHS_DICT.items():
            sounds[key] = pygame.mixer.Sound(value)
        # 导入所有图片
        images = {}
        for key, value in cfg.IMAGE_PATHS_DICT.items():
            if isinstance(value, list):
                images[key] = []
                for imagepath in value:
                    images[key].append(pygame.image.load(imagepath))
            else:
                images[key] = pygame.image.load(value)
        # AIAgent
        ai_agent = AIAgent(images, sounds)
        # 最高分
        highest_score, flag, num_iterations = 0, True, 0
        # 游戏轮次切换
        while flag:
            # 定义一些游戏中必要的元素和变量
            score, highest_score = 0, highest_score
            score_board = Scoreboard(images['numbers'], position=(534, 15), bg_color=cfg.BACKGROUND_COLOR)
            highest_score_board = Scoreboard(images['numbers'], position=(435, 15), bg_color=cfg.BACKGROUND_COLOR, is_highest=True)
            ground = Ground(images['ground'], position=(0, cfg.SCREENSIZE[1]))
            cloud_sprites_group, cactus_sprites_group, ptera_sprites_group = pygame.sprite.Group(), pygame.sprite.Group(), pygame.sprite.Group()
            add_obstacle_timer, score_timer = 0, 0
            # 游戏主循环
            clock = pygame.time.Clock()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                screen.fill(cfg.BACKGROUND_COLOR)
                # --自动决定每个小恐龙的动作
                if len(cactus_sprites_group) > 0 or len(ptera_sprites_group) > 0:
                    nearest_obstacle = None
                    for item in cactus_sprites_group:
                        if item.rect.left < 84:
                            continue
                        if nearest_obstacle is None:
                            nearest_obstacle = item
                        else:
                            if item.rect.left < nearest_obstacle.rect.left: nearest_obstacle = item
                    for item in ptera_sprites_group:
                        if item.rect.left < 84:
                            continue
                        if nearest_obstacle is None:
                            nearest_obstacle = item
                        else:
                            if item.rect.left < nearest_obstacle.rect.left: nearest_obstacle = item
                    if nearest_obstacle:
                        inputs = [nearest_obstacle.rect.left-84, nearest_obstacle.rect.bottom, nearest_obstacle.rect.width, nearest_obstacle.rect.height, -ground.speed]
                        ai_agent.decide(inputs)
                # --随机添加云
                if len(cloud_sprites_group) < 5 and random.randrange(0, 300) == 10:
                    cloud_sprites_group.add(Cloud(images['cloud'], position=(cfg.SCREENSIZE[0], random.randrange(30, 75))))
                # --随机添加仙人掌/飞龙
                add_obstacle_timer += 1
                if add_obstacle_timer > random.randrange(50, 150):
                    add_obstacle_timer = 0
                    random_value = random.randrange(0, 10)
                    if random_value >= 5 and random_value <= 7:
                        cactus_sprites_group.add(Cactus(images['cacti']))
                    else:
                        position_ys = [cfg.SCREENSIZE[1]*0.82, cfg.SCREENSIZE[1]*0.75, cfg.SCREENSIZE[1]*0.60, cfg.SCREENSIZE[1]*0.20]
                        ptera_sprites_group.add(Ptera(images['ptera'], position=(600, random.choice(position_ys))))
                # --更新游戏元素
                ai_agent.update()
                ground.update()
                cloud_sprites_group.update()
                cactus_sprites_group.update()
                ptera_sprites_group.update()
                score_timer += 1
                if score_timer > (cfg.FPS//12):
                    score_timer = 0
                    score += 1
                    score = min(score, 99999)
                    if score > highest_score:
                        highest_score = score
                    if score % 100 == 0:
                        sounds['point'].play()
                    if score % 1000 == 0:
                        ground.speed -= 1
                        for item in cloud_sprites_group:
                            item.speed -= 1
                        for item in cactus_sprites_group:
                            item.speed -= 1
                        for item in ptera_sprites_group:
                            item.speed -= 1
                # --碰撞检测
                for cacti in cactus_sprites_group:
                    for dino in ai_agent.dinos.values():
                        if dino['sprite'].is_dead: continue
                        if pygame.sprite.collide_mask(dino['sprite'], cacti):
                            dino['sprite'].die(sounds)
                for ptera in ptera_sprites_group:
                    for dino in ai_agent.dinos.values():
                        if dino['sprite'].is_dead: continue
                        if pygame.sprite.collide_mask(dino['sprite'], ptera):
                            dino['sprite'].die(sounds)
                # --将游戏元素画到屏幕上
                ai_agent.draw(screen)
                ground.draw(screen)
                cloud_sprites_group.draw(screen)
                cactus_sprites_group.draw(screen)
                ptera_sprites_group.draw(screen)
                score_board.set(score)
                highest_score_board.set(highest_score)
                score_board.draw(screen)
                highest_score_board.draw(screen)
                # --更新屏幕
                pygame.display.update()
                clock.tick(cfg.FPS)
                # --更新模型
                num_dies, num_alives = 0, 0
                for dino in ai_agent.dinos.values():
                    if dino['sprite'].is_dead:
                        num_dies += 1
                    else:
                        num_alives += 1
                        dino['sprite'].score = score
                self.logging(f'Iteration: {num_iterations}, Score: {score}, Max Score: {highest_score}, Dead dino: {num_dies}, Alive dino: {num_alives}')
                if num_alives == 0: break
            ai_agent.nextgeneration()
    '''logging'''
    def logging(self, msg, tip='INFO'):
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {tip}]: {msg}')


'''run'''
if __name__ == '__main__':
    client = TRexRushGame()
    client.run()