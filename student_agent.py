import cv2
import gym
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import random


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
        self.net2 = CNNDQN((4, 84, 84), 7)
        # self.net.load_state_dict(torch.load("/Users/ilo/Code/DRL_HW3/DRL-Assignment-3/super-mario-bros-dqn/recording3/run40/SuperMarioBros-v0.dat", map_location=torch.device("cpu")))
        self.net.load_state_dict(torch.load("./SuperMarioBros-v0.dat", map_location=torch.device("cpu")))
        # self.net2.load_state_dict(torch.load("./pretrained2.dat", map_location=torch.device("cpu")))
        self.net.eval()

        # Frame buffer = (4, 84, 84)
        self.buffer = np.zeros((4, 84, 84), dtype=np.float32)
        self.obs_buffer = deque(maxlen=2)
        self.skip_counter = 0
        self.last_action = None
        self.epsilon = 0.01

    def preprocess_frame(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)
    
    
    def update_frame_buffer(self, frame):
        # Slide frames left, insert new one at the end
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame

    def act(self, obs):
        # Simulate max pooling over last 2 raw frames
        if len(self.obs_buffer) < 2 and self.last_action == None :
            # FrameDownsample:
            obs = self.preprocess_frame(obs)
            
            #image to pytorch
            obs =np.moveaxis(obs, 2, 0)

            # Update frame buffer
            self.update_frame_buffer(obs)

            # Normalize to float
            obs = np.array(self.buffer).astype(np.float32) / 255.0

            state_v = torch.tensor(np.array([obs], copy=False))
            q_vals = self.net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            self.last_action = action
            self.skip_counter = 3
            return action
        elif self.skip_counter > 0:
            self.skip_counter -= 1
            self.obs_buffer.append(obs)
            return self.last_action
        else:
            self.obs_buffer.append(obs)
            obs = np.maximum(self.obs_buffer[-1], self.obs_buffer[-2])


            # FrameDownsample:
            obs = self.preprocess_frame(obs)
            
            #image to pytorch
            obs =np.moveaxis(obs, 2, 0)

            # Update frame buffer
            self.update_frame_buffer(obs)

            # Normalize to float
            obs = np.array(self.buffer).astype(np.float32) / 255.0

            state_v = torch.tensor(np.array([obs], copy=False))
            
            q_vals = self.net(state_v).data.numpy()[0]
            # q_vals2 = self.net2(state_v).data.numpy()[0]
            # print(q_vals)
            action1 = np.argmax(q_vals)
            # action2 = np.argmax(q_vals2)
            action = action1
            
            # if self.epsilon > 0.2:
            #     action = action2
            #     if random.random() < 0.1:
            #         action = random.randint(0, 11)
            #     self.skip_counter = 20
            # else:
            #     self.epsilon *= 1.0095
            self.skip_counter = 3
            
            # if random.random() < self.epsilon:
            #     if self.epsilon > 0.2:
            #         action = random.randint(0, 11)
            #         self.skip_counter = 40
            # else:
            #     self.epsilon *= 1.0095
            
            # if random.random() < 0.01:
            #     action = random.randint(0, 11)
            #     self.skip_counter = 4

                
            self.last_action = action
            
                
            # if self.epsilon > 0.5:
            #     self.epsilon = 0.01
            # print(self.epsilon)
            
            
            return action

# class Agent:
#     def __init__(self):
#         self.action_space = gym.spaces.Discrete(12)
#         self.net = CNNDQN((4, 84, 84), 12)
#         # self.net.load_state_dict(torch.load("/Users/ilo/Code/DRL_HW3/DRL-Assignment-3/super-mario-bros-dqn/recording/run1250/SuperMarioBros-1-1-v0.dat", map_location=torch.device("cpu")))
#         self.net.load_state_dict(torch.load("./pretrained.dat", map_location=torch.device("cpu"))) # best repeat is 2
#         self.net.eval()

#         # Frame buffer = (4, 84, 84)
#         self.buffer = np.zeros((4, 84, 84), dtype=np.uint8)
#         self.obs_buffer = deque(maxlen=2)
#         self.skip_counter = 0
#         self.last_action = 0

#     def preprocess_frame(self, obs):
#         gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
#         return resized[:, :, None].astype(np.uint8)

#     def update_frame_buffer(self, frame):
#         # Slide frames left, insert new one at the end
#         self.buffer[:-1] = self.buffer[1:]
#         self.buffer[-1] = frame

#     def act(self, obs):
#         # Max and Skip
#         self.obs_buffer.append(obs)
#         max_frame = np.max(np.stack(self.obs_buffer), axis=0)

#         # FrameDownsample:
#         processed = self.preprocess_frame(max_frame)
        
#         #Image to pytorch
#         tensor = np.moveaxis(processed, 2, 0)

       
        
#         if not hasattr(self, 'initialized'):
#             for i in range(4):
#                 self.buffer[i] = tensor
#             self.initialized = True
#         else:
#             # Update frame buffer: (4, 84, 84)
#             self.update_frame_buffer(tensor)

#         # Normalize to float
#         input_tensor = torch.tensor([self.buffer], dtype=torch.float32) / 255.0
        
#         # Action repeat logic
#         if self.skip_counter > 0:
#             self.skip_counter -= 1
#             return self.last_action

#         with torch.no_grad():
#             q_vals = self.net(input_tensor)[0].numpy()

#         action = int(np.argmax(q_vals))
#         self.last_action = action
#         self.skip_counter = 3
#         # if random.random() < 0.01:
#         #     action = random.randint(0, 11)
#         return action

# import cv2
# import gym
# import torch
# import numpy as np
# from collections import deque
# import torch.nn as nn


# class CNNDQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(CNNDQN, self).__init__()
#         self._input_shape = input_shape
#         self._num_actions = num_actions

#         self.features = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(self.feature_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_actions)
#         )

#     def forward(self, x):
#         x = self.features(x).view(x.size(0), -1)
#         return self.fc(x)

#     @property
#     def feature_size(self):
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *self._input_shape)
#             return self.features(dummy_input).view(1, -1).size(1)


# class Agent:
#     def __init__(self):
#         self.action_space = gym.spaces.Discrete(12)
#         self.net = CNNDQN((4, 84, 84), 12)
#         # self.net.load_state_dict(torch.load("/Users/ilo/Code/DRL_HW3/DRL-Assignment-3/super-mario-bros-dqn/recording/run1598/SuperMarioBros-1-1-v0.dat", map_location=torch.device("cpu")))
# #         
#         self.net.load_state_dict(torch.load("pretrained.dat", map_location=torch.device("cpu")))
#         self.net.eval()

#         self.obs_buffer = deque(maxlen=2)                # raw observations for max pooling
#         self.frame_buffer = np.zeros((4, 84, 84), dtype=np.uint8)  # stacked preprocessed frames
#         self.skip_counter = 0                            # for action repeat
#         self.last_action = None
#         self.initialized = False

#     def preprocess(self, obs):
#         gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
#         return resized.astype(np.uint8)

#     def act(self, obs):
#         if not self.initialized:
#             for i in range(4):
#                 #initialize as zero
#                 self.frame_buffer[i] = np.zeros((84, 84), dtype=np.uint8)
#             self.initialized = True
#         # else:
#         #     self.frame_buffer[:-1] = self.frame_buffer[1:]
#         #     self.frame_buffer[-1] = frame
#         # -- Step 1: Max pool incoming obs --
#         # self.obs_buffer.append(obs)
#         if len(self.obs_buffer) < 2 and self.last_action == None :
#             input_tensor = torch.tensor([self.frame_buffer], dtype=torch.float32) / 255.0
#             with torch.no_grad():
#                 q_vals = self.net(input_tensor)[0].numpy()
#             action = int(np.argmax(q_vals))

#             self.last_action = action
#             self.skip_counter = 3
#             return action
        
#         if self.skip_counter > 0:
#             self.skip_counter -= 1
#             self.obs_buffer.append(obs)
#             return self.last_action
        
#         obs = np.maximum(self.obs_buffer[-1], self.obs_buffer[-2])

#         # -- Step 2: Preprocess frame --
#         frame = self.preprocess(obs)

#         # -- Step 3: Update frame buffer for stacking --
#         if not self.initialized:
#             for i in range(4):
#                 self.frame_buffer[i] = frame
#             self.initialized = True
#         else:
#             self.frame_buffer[:-1] = self.frame_buffer[1:]
#             self.frame_buffer[-1] = frame

#         # -- Step 5: Q-value inference --
#         input_tensor = torch.tensor([self.frame_buffer], dtype=torch.float32) / 255.0
#         with torch.no_grad():
#             q_vals = self.net(input_tensor)[0].numpy()
#         action = int(np.argmax(q_vals))

#         self.last_action = action
#         self.skip_counter = 3
#         if random.random() < 0.03:
#             action = random.randint(0, 11)
#         return action
