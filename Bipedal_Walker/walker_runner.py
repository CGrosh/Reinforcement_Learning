import torch
import torch.nn as nn 
import numpy as np 
import gym 
gym.logger.set_level(40)
from a2c_walker import *
from torch import optim 
from torch.distributions.normal import Normal 
import matplotlib.pyplot as plt 

NUM_EPS = 10000
GAMMA = 0.9898
ALPHA = 0.1

env = gym.make("BipedalWalker-v3")

ac_mod = a2c_walker(obs_dim=24, act_dim=4, hidden1=64, \
    hidden2=124, hidden3=188, hidden4=124, drop_rate=0.2)
optimizer = optim.Adam(ac_mod.parameters(), lr=0.005)
critic_mse = nn.MSELoss()

total_loss = []
total_reward = []
ep_track = []

for ep in range(NUM_EPS):
    obs = env.reset()
    env.render()
    cond = True 

    log_probs = []
    values = []
    ep_rewards = []
    
    while cond == True:
        mu, var = ac_mod.actor_forward(obs)
        dist = Normal(mu, torch.sqrt(var))
        action = dist.sample().clamp(-1,1)

        obs_new, reward, done, info = env.step(action.numpy())

        q_val = ac_mod.critic_forward(obs, action.numpy())
        q_val = q_val.detach().numpy()

        log_prob = dist.log_prob(action)
        env.render()

        ep_rewards.append(reward)
        log_probs.append(log_prob)
        values.append(q_val)
        obs = obs_new 

        if done == True:
            mu, var = ac_mod.actor_forward(obs_new)
            dist = Normal(mu, torch.sqrt(var))
            action = dist.sample().clamp(-1,1)  

            q_val = ac_mod.critic_forward(obs_new, action.numpy())
            q_val = q_val.detach().numpy()
            cond = False

    qval_buffer = np.zeros_like(values)
    for i in reversed(range(len(ep_rewards))):
        q_target = (1-ALPHA)*ep_rewards[i] + (GAMMA*q_val)
        qval_buffer[i] = q_target 

    values, qvals = torch.FloatTensor(values), \
        torch.FloatTensor(qval_buffer)
    log_probs = torch.stack(log_probs)

    advantage = qvals - values 
    actor_loss = (-log_probs * advantage).mean()

    critic_loss = critic_mse(values, qvals)
    entropy_loss = dist.entropy().mean() * 0.001

    ac_loss = actor_loss + critic_loss + entropy_loss 

    total_loss.append(ac_loss.item())
    total_reward.append(np.mean(ep_rewards))
    ep_track.append(ep)

    optimizer.zero_grad()
    ac_loss.backward()
    optimizer.step()

    print("Episode {} : {}".format(ep, np.mean(ep_rewards)))

fig, loss_graph = plt.subplots(1)
fig2, reward_graph = plt.subplots(1)

loss_graph.plot(total_loss)
reward_graph.plot(total_reward)

fig.show()
fig2.show()
plt.show()



        
