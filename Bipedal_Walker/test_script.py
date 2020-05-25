import torch.nn as nn 
import torch 
import numpy as np 
import gym 
gym.logger.set_level(40)
from a2c_walker import *
from torch.distributions.normal import Normal 

env = gym.make("BipedalWalker-v3")
obs = env.reset()

mod = a2c_walker(obs_dim=24, act_dim=4, hidden1=64, hidden2=124, \
    hidden3=188, hidden4=124, drop_rate=0.2)

mu, var = mod.actor_forward(obs)
dist = Normal(mu, torch.sqrt(var))
action = dist.sample().clamp(-1,1)

print(action)
print(dist.log_prob(action))