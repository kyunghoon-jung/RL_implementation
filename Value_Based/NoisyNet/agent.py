import gym
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torchsummary import summary

from qnetwork import * 
from replay_buffer import ReplayBuffer
from agent_utils import Agent_Utils

import wandb

class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_frame: ('int: the number of channels of input image'),
                 input_dim: ('int: the width and height of pre-processed input image'),
                 num_frames: ('int: Total number of frames'),
                 skipped_frame: ('int: The number of skipped frames'),
                 gamma: ('float: Discount Factor'),
                 eps_max: ('float: Max epsilon'),
                 eps_decay: ('float: Amount of epsilon decay per action'),
                 eps_min: ('float: Min epsilon'),
                 current_update_freq: ('int: Current Q Network Update Frequency (by frames)'),
                 target_update_freq: ('int: Target Q Network Update Frequency (by frames)'),
                 is_noisy: ('str: whether NoisyNet is applied or not'),
                 input_feature_type: ('str: 1dim or 2dim'),
                 initial_std: ('float: noise standard deviation')=0.5, # 0.5 is the default value in the paper
                 update_type: ('str: Update type for target network. Hard or Soft')='hard',
                 soft_update_tau: ('float: Soft update ratio')=None,
                 batch_size: ('int: Update batch size')=32,
                 buffer_size: ('int: Replay buffer size')=1000000,
                 update_start_buffer_size: ('int: Update starting buffer size')=50000,
                 learning_rate: ('float: Learning rate')=0.0004,
                 device_num: ('int: GPU device number')=0,
                 rand_seed: ('int: Random seed')=None,
                 plot_option: ('str: Plotting option')=False,
                 model_path: ('str: Model saving path')='./',
                 trained_model_path: ('str: Trained model path')=None):
        '''NoisyNet: epsilon greedy selection is not applied as in the paper.
                     So there isn't any variables related to it, but added 'init_noise_std' variable.'''

        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.env = env
        self.input_frames = input_frame
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.is_noisy = is_noisy
        self.input_feature_type = input_feature_type
        self.initial_std = initial_std        
        self.skipped_frame = skipped_frame
        self.epsilon = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.current_update_freq = current_update_freq
        self.target_update_freq = target_update_freq
        self.current_update_cnt = 0
        self.target_update_cnt = 0
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option

        self.agent_utils = Agent_Utils(self)

        if trained_model_path != None:
            self.agent_utils.q_current.load_state_dict(
                torch.load(trained_model_path)
            )
            print("Pre-trained model is loaded successfully.")
        self.agent_utils.q_target.load_state_dict(self.agent_utils.q_current.state_dict())
        self.agent_utils.q_target.eval()
        self.optimizer = optim.Adam(self.agent_utils.q_current.parameters(), lr=learning_rate) 

    def store(self, state, action, reward, next_state, done):
        self.agent_utils.memory.store(state, action, reward, next_state, done)

    def update_current_q_net(self):
        batch = self.agent_utils.memory.batch_load()
        loss = self.agent_utils._compute_loss(self, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def target_soft_update(self):
        for target_param, current_param in zip(self.agent_utils.q_target.parameters(), self.agent_utils.q_current.parameters()):
            target_param.data.copy_(self.tau*current_param.data + (1.0-self.tau)*target_param.data)

    def target_hard_update(self):
        self.agent_utils.q_target.load_state_dict(self.agent_utils.q_current.state_dict())

    def train(self):
        tic = time.time()

        print("Storing initial buffer before starting to learn..")
        state = self.agent_utils.get_init_state(self)
        for frame_idx in range(1, self.update_start+1):
            _, action = self.agent_utils.select_action(self, state)
            self.agent_utils.q_current.init_noise()
            reward, next_state, done = self.agent_utils.get_state(self, state, action)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)
            state = next_state
            if done: state = self.agent_utils.get_init_state(self)

        print("Done. Start learning..")
        score = 0
        epi_Qs = []
        epi_loss = []
        total_losses = []
        history_store = []
        scores = [-100000]
        avg_scores = [-100000]
        
        for frame_idx in range(1, self.num_frames+1):
            Qs, action = self.agent_utils.select_action(self, state)
            reward, next_state, done = self.agent_utils.get_state(self, state, action)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)
            history_store.append([state[0], Qs, action, reward, next_state[0], done])

            self.current_update_cnt = (self.current_update_cnt+1) % self.current_update_freq 
            self.target_update_cnt = (self.target_update_cnt+1) % self.target_update_freq
            if self.current_update_cnt == 0:
                loss = self.update_current_q_net()
                epi_loss.append(loss)
                if self.update_type=='soft': self.target_soft_update()
                elif self.target_update_cnt==0: self.target_hard_update()
                self.agent_utils.q_current.init_noise()
                self.agent_utils.q_target.init_noise()

            score += reward
            epi_Qs.append(Qs)
            if done:
                scores.append(score)
                self.agent_utils.eval_and_save(self, frame_idx, history_store, scores, avg_scores, tic)
                avg_scores.append(np.mean(scores[-10:]))
                self.agent_utils.plot(self, frame_idx, score, scores, total_losses, epi_Qs, epi_loss)
                score = 0
                epi_Qs = []
                epi_loss = []
                state = self.agent_utils.get_init_state(self)
                history_store = []
            else: state = next_state

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))
