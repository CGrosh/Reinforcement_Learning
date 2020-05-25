from pg_mod import *
import torch
import torch.nn as nn 
from torch import optim 
import numpy as np 
import matplotlib.pylot as plt 
import time 
import gym 
gym.logger.set_level(40)

mod = CustomModel(obs_dim=4, act_dim=2, hidden1=128, \
    hidden2=192, hidden3=64, drop_rate=0.2)

env = gym.make("CartPole-v1")
obs = env.reset()

N_EPS = 100
lr = 0.005
optimizer = optim.Adam(mod.parameters(), lr=lr)
compute_loss = nn.NLLLoss()

