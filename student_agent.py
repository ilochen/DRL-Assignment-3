import gym
import torch
import numpy as np
from random import random, randrange

import torch.nn as nn

# Do not modify the input of the 'act' function and the '__init__' function. 
class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

    def act(self, state, epsilon, device):
        if random() > epsilon:
            state = torch.FloatTensor(np.float32(state)) \
                .unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(self._num_actions)
        return action


class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.net = CNNDQN((4, 84, 84), 12)
        self.net.load_state_dict(torch.load("./pretrained.dat", map_location=torch.device('cpu')))

    def act(self, observation):
        state_v = torch.tensor(np.array([observation], copy=False))
        q_vals = self.net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        return action
        # return self.action_space.sample()