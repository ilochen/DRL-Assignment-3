import cv2
import gym
import torch
import numpy as np
from collections import deque
from random import random, randrange
import torch.nn as nn


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
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.net = CNNDQN((4, 84, 84), 12)
        self.net.load_state_dict(torch.load("./pretrained.dat", map_location=torch.device('cpu')))
        self.frame_stack = deque(maxlen=4)

    def preprocess_observation(self, obs):
        # Convert RGB to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # Reshape to (1, 84, 84)
        obs = np.expand_dims(obs, axis=0)
        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        return obs

    def act(self, observation):
        processed = self.preprocess_observation(observation)
        self.frame_stack.append(processed)

        while len(self.frame_stack) < 4:
            self.frame_stack.appendleft(processed.copy())

        state = np.concatenate(self.frame_stack, axis=0)  # Shape: (4, 84, 84)
        state_v = torch.tensor(np.array([state]), dtype=torch.float32)
        q_vals = self.net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        return action
