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

from qnetwork import QNetwork 
from replay_buffer import ReplayBuffer

import wandb

class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_frame: ('int: the number of channels of input image'),
                 input_dim: ('int: the width and height of pre-processed input image'),
                 num_frames: ('int: Total number of training frames'),
                 skipped_frame: ('int: The number of skipped frames in the environment'),
                 eps_decay: ('float: Epsilon Decay_rate'),
                 gamma: ('float: Discount Factor'),
                 target_update_freq: ('int: Target Update Frequency (by frames)'),
                 update_type: ('str: Update type for target network. Hard or Soft')='hard',
                 soft_update_tau: ('float: Soft update ratio')=None,
                 batch_size: ('int: Update batch size')=32,
                 buffer_size: ('int: Replay buffer size')=1000000,
                 update_start_buffer_size: ('int: Update starting buffer size')=50000,
                 learning_rate: ('float: Learning rate')=0.0004,
                 eps_min: ('float: Epsilon Min')=0.1,
                 eps_max: ('float: Epsilon Max')=1.0,
                 device_num: ('int: GPU device number')=0,
                 rand_seed: ('int: Random seed')=None,
                 plot_option: ('str: Plotting option')=False,
                 model_path: ('str: Model saving path')='./',
                 trained_model_path: ('str: Trained model path')=''):

        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        self.env = env
        self.input_frames = input_frame
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.skipped_frame = skipped_frame
        self.epsilon = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_cnt = 0
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option
        
        self.q_behave = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        self.q_target = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        if trained_model_path:
            self.q_behave.load_state_dict(torch.load(trained_model_path))
            print("Trained model is loaded successfully.")
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 

        self.memory = ReplayBuffer(self.buffer_size, (self.input_frames, self.input_dim, self.input_dim), self.batch_size)

    def select_action(self, state: 'Must be pre-processed in the same way as updating current Q network. See def _compute_loss'):
        
        if np.random.random() < self.epsilon:
            return np.zeros(self.action_dim), self.env.action_space.sample()
        else:
            # Normalize를 위해서 255로 나누어줍니다. 
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)/255
            Qs = self.q_behave(state)
            action = Qs.argmax()
            return Qs.detach().cpu().numpy(), action.detach().item()

    def processing_resize_and_gray(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Gray scale로 바꿔줍니다
        frame = cv2.resize(frame, dsize=(self.input_dim, self.input_dim)).reshape(self.input_dim, self.input_dim).astype(np.uint8) # np.uint8로 바꿔주어, 저장 용량을 최소화합니다.
        return frame 

    def get_init_state(self):

        init_state = np.zeros((self.input_frames, self.input_dim, self.input_dim))
        init_frame = self.env.reset()
        init_state[0] = self.processing_resize_and_gray(init_frame)
        
        for i in range(1, self.input_frames): # Q network에 들어가는 frame의 수 만큼 반복합니다.
            action = self.env.action_space.sample()
            for j in range(self.skipped_frame): # skipped frame 수 만큼 동일한 action을 넣어줍니다.
                state, _, _, _ = self.env.step(action) 
            state, _, _, _ = self.env.step(action) 
            init_state[i] = self.processing_resize_and_gray(state) 
        return init_state

    def get_state(self, state, action, skipped_frame=0):
        '''
        input_frames: how many frames to be merged
        input_dim: hight and width of input resized image
        skipped_frame: how many frames to be skipped
        '''
        next_state = np.zeros((self.input_frames, self.input_dim, self.input_dim))
        for i in range(len(state)-1):
            next_state[i] = state[i+1]

        rewards = 0
        dones = 0
        for _ in range(skipped_frame-1):
            state, reward, done, _ = self.env.step(action) 
            rewards += reward 
            dones += int(done) 
        state, reward, done, _ = self.env.step(action) 
        next_state[-1] = self.processing_resize_and_gray(state) 
        rewards += reward 
        dones += int(done) 
        return rewards, next_state, dones

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def update_current_q_net(self):
        batch = self.memory.batch_load()
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # Soft Update 방식으로 target network 업데이트하는 함수
    def target_soft_update(self):
        for target_param, current_param in zip(self.q_target.parameters(), self.q_behave.parameters()):
            target_param.data.copy_(self.tau*current_param.data + (1.0-self.tau)*target_param.data)

    # Hard Update 방식으로 target network 업데이트하는 함수
    def target_hard_update(self):
        self.update_cnt = (self.update_cnt+1) % self.target_update_freq
        if self.update_cnt==0:
            self.q_target.load_state_dict(self.q_behave.state_dict())

    def train(self):
        tic = time.time()
        losses = []
        scores = []
        epsilons = []
        avg_scores = [[-1000]]

        score = 0

        print("Storing initial buffer..")  # 먼저 Buffer를 일정 정도 쌓는 반복문입니다.
        state = self.get_init_state()
        for frame_idx in range(1, self.update_start+1):
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            self.store(state, action, reward, next_state, done)
            state = next_state
            if done: state = self.get_init_state()

        print("Done. Start learning..")
        history_store = []
        for frame_idx in range(1, self.num_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            self.store(state, action, reward, next_state, done)
            history_store.append([state, Qs, action, reward, next_state, done]) # history_store는 차후에 어떻게 환경에서 행동했는지 뽑아보기 위한 변수입니다.
            loss = self.update_current_q_net()

            if self.update_type=='hard':   self.target_hard_update()
            elif self.update_type=='soft': self.target_soft_update()
            
            score += reward
            losses.append(loss)

            if done: # done state 일때, 조건문에 따라 모델을 저장하고, 결과를 plotting합니다.
                scores.append(score)
                if np.mean(scores[-10:]) > max(avg_scores):
                    torch.save(self.q_behave.state_dict(), self.model_path+'{}_Score:{}.pt'.format(frame_idx, np.mean(scores[-10:])))
                    training_time = round((time.time()-tic)/3600, 1)
                    np.save(self.model_path+'{}_history_Score_{}_{}hrs.npy'.format(frame_idx, score, training_time), np.array(history_store))
                    print("          | Model saved. Recent scores: {}, Training time: {}hrs".format(scores[-10:], training_time), ' /'.join(os.getcwd().split('/')[-3:]))
                avg_scores.append(np.mean(scores[-10:]))

                if self.plot_option=='inline': # 쥬피터 노트북에서 돌릴 수 있는 코드
                    scores.append(score)
                    epsilons.append(self.epsilon)
                    self._plot(frame_idx, scores, losses, epsilons)
                elif self.plot_option=='wandb': 
                    wandb.log({'Score': score, 'loss(10 frames avg)': np.mean(losses[-10:]), 'Epsilon': self.epsilon})
                    print(score, end='\r')
                else: 
                    print(score, end='\r')

                score=0
                state = self.get_init_state()
                history_store = []
            else: state = next_state

            self._epsilon_step()

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))

    def _epsilon_step(self):
        ''' Epsilon decay control '''
        eps_decay_list = [self.eps_decay, self.eps_decay/2.5, self.eps_decay/3.5, self.eps_decay/5.5] 

        if self.epsilon>0.30:
            self.epsilon = max(self.epsilon-eps_decay_list[0], 0.1)
        elif self.epsilon>0.25:
            self.epsilon = max(self.epsilon-eps_decay_list[1], 0.1)
        elif self.epsilon>1.7:
            self.epsilon = max(self.epsilon-eps_decay_list[2], 0.1)
        else:
            self.epsilon = max(self.epsilon-eps_decay_list[3], 0.1)

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        # If normalization is used, it must be applied to 'state' and 'next_state' here. ex) state/255
        states = torch.FloatTensor(batch['states']).to(self.device) / 255
        next_states = torch.FloatTensor(batch['next_states']).to(self.device) / 255
        actions = torch.LongTensor(batch['actions'].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)

        current_q = self.q_behave(states).gather(1, actions)

        next_q = self.q_target(next_states).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - dones
        target = (rewards + (mask * self.gamma * next_q)).to(self.device)

        loss = F.smooth_l1_loss(current_q, target)
        return loss

    def _plot(self, frame_idx, scores, losses, epsilons):
        clear_output(True) 
        plt.figure(figsize=(20, 5), facecolor='w') 
        plt.subplot(131)  
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores) 
        plt.subplot(132) 
        plt.title('loss') 
        plt.plot(losses) 
        plt.subplot(133) 
        plt.title('epsilons')
        plt.plot(epsilons) 
        plt.show() 

if __name__=='__main__':
    agent = Agent()