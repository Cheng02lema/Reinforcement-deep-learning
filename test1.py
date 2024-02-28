import time

import gym   # 导入gym库
from gym.error import ResetNeeded
import matplotlib 
import pygame
env = gym.make('CartPole-v1', render_mode="human")    #创建小车倒立模型的环境

env.reset()  #  初始化环境

env.render()   # 显示当前环境   需要提前安装matplotlib包
time.sleep(100)