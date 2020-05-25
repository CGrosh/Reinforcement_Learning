import torch 
import torch.nn as nn 
import numpy as np 
from torch import optim 

class CustomModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden1, \
        hidden2, hidden3, drop_rate):
        super(CustomModel, self).__init__()

        # Initialize Variables 
        self.obs_dim = obs_dim
        self.act_dim = act_dim 
        self.hidden1 = hidden1 
        self.hidden2 = hidden2 
        self.hidden3 = hidden3 
        self.drop_rate = drop_rate

        # Initialize the basic input and hidden layers of the network 
        self.layer_one = nn.Linear(self.obs_dim, self.hidden1)
        self.Dropout_layer = nn.Dropout(p=self.drop_rate)
        self.layer_two = nn.Linear(self.hidden1, self.hidden2)
        self.layer_three = nn.Linear(self.hidden2, self.hidden3)

        # Actor output layer 
        self.act_out = nn.Linear(self.hidden3, self.act_dim)

       # The critit net output layer 
        self.critic_out = nn.Linear(self.hidden3, 1)

    def forward(self, obs):
        # Format input and build softmax and activation 
        obs = torch.from_numpy(obs).float()
        softer = nn.Softmax()
        softplus = nn.Softplus()

        # Run observation through the input and hidden layers 
        init_input = torch.relu(self.layer_one(obs))
        dropper = self.Dropout_layer(init_input)
        hidden_one = torch.relu(self.layer_two(dropper))
        hidden_two  = torch.relu(self.layer_three(hidden_one))

        # Actor output 
        act_out = torch.tanh(self.act_out(hidden_two))

        # Critic output 
        crit_out = 


        return output 



