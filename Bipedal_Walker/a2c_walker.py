import torch 
import torch.nn as nn 
import numpy as np 


class a2c_walker(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden1, hidden2, \
        hidden3, hidden4, drop_rate):
        super(a2c_walker, self).__init__()

        # Initialize input parameters 
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.hidden1 = hidden1 
        self.hidden2 = hidden2 
        self.hidden3 = hidden3 
        self.hidden4 = hidden4 
        self.drop_rate = drop_rate 

        # Actor Input
        self.obs_input = nn.Linear(114, self.hidden1)

        # Critic Input
        self.critic_input = nn.Linear(118, self.hidden1)
        
        # Hidden layers 
        self.hid1 = nn.Linear(self.hidden1, self.hidden2)
        self.hid2 = nn.Linear(self.hidden2, self.hidden3)
        self.hid3 = nn.Linear(self.hidden3, self.hidden4)

        # Actor Mean Output 
        self.mu_out = nn.Linear(self.hidden4, self.act_dim)
        # Actor Sigma Output 
        self.sigma_out = nn.Linear(self.hidden4, self.act_dim)

        # Critic Q-Val Output 
        self.critic_output = nn.Linear(self.hidden4, 1)

        # Lidar Ray Convolutional Network 
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=25, \
                kernel_size=3, stride=1), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2), 
            nn.Flatten()
        )
    
    def actor_forward(self, obs):
        lidar = torch.from_numpy(obs[14:]).float()
        pos_input = torch.from_numpy(obs[:14]).float()

        lidar_input = lidar.view(1,1,10)
        conv_lidar = self.conv_net(lidar_input)

        full_input = torch.cat((pos_input, conv_lidar[0]), dim=0)

        x = torch.relu(self.obs_input(full_input))
        first_hid = torch.relu(self.hid1(x))
        sec_hid = torch.relu(self.hid2(first_hid))
        third_hid = torch.relu(self.hid3(sec_hid))

        softer = nn.Softplus()
        
        obs_mu = torch.tanh(self.mu_out(third_hid))
        obs_sigma = softer(self.sigma_out(third_hid))

        return obs_mu, obs_sigma
    
    def critic_forward(self, obs, action):

        action = torch.from_numpy(action).float()
        lidar = torch.from_numpy(obs[14:]).float()
        pos_input = torch.from_numpy(obs[:14]).float()

        lidar_input = lidar.view(1,1,10)
        conv_lidar = self.conv_net(lidar_input)

        full_input = torch.cat((pos_input, conv_lidar[0]), dim=0)
        sa_input = torch.cat((full_input, action), dim=0)
        
        x = torch.relu(self.critic_input(sa_input))
        first_hid = torch.relu(self.hid1(x))
        sec_hid = torch.relu(self.hid2(first_hid))
        third_hid = torch.relu(self.hid3(sec_hid))

        q_val = self.critic_output(third_hid)
        return q_val 

