import cv2
import gym
import torch
import numpy as np
from collections import deque
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
        x = self.features(x).view(x.size(0), -1)
        return self.fc(x)

    @property
    def feature_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self._input_shape)
            return self.features(dummy_input).view(1, -1).size(1)


class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.net = CNNDQN((4, 84, 84), 12)
        self.net.load_state_dict(torch.load("pretrained.dat", map_location=torch.device("cpu")))
        self.net.eval()

        # Frame buffer = (4, 84, 84)
        self.buffer = np.zeros((4, 84, 84), dtype=np.float32)
        self.obs_buffer = deque(maxlen=2)
        self.skip_counter = 0
        self.last_action = 0

    def preprocess_frame(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32)

    def update_frame_buffer(self, frame):
        # Slide frames left, insert new one at the end
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame

    def act(self, obs):
        # Simulate max pooling over last 2 raw frames
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) == 2:
            obs = np.maximum(self.obs_buffer[0], self.obs_buffer[1])
        else:
            obs = self.obs_buffer[0]

        # Preprocess: grayscale + resize (returns (84, 84))
        processed = self.preprocess_frame(obs)

        # Update frame buffer
        self.update_frame_buffer(processed)

        # Expand dims to (1, 4, 84, 84) and normalize
        input_tensor = torch.tensor([self.buffer], dtype=torch.float32) / 255.0

        # Action repeat
        if self.skip_counter > 0:
            self.skip_counter -= 1
            return self.last_action

        with torch.no_grad():
            q_vals = self.net(input_tensor)[0].numpy()

        action = int(np.argmax(q_vals))
        self.last_action = action
        self.skip_counter = 2
        return action
