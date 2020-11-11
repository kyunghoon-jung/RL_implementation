''' Dueling + Multi_Step TD Agent '''
import gym
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torchsummary import summary

from qnetwork import QNetwork 
from replay_buffer import ReplayBuffer

import wandb
import visdom

class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_frame: ('int: the number of channels of input image'),
                 input_dim: ('int: the width and height of pre-processed input image'),
                 num_frames: ('int: Total number of frames'),
                 skipped_frame: ('int: The number of skipped frames'),
                 eps_decay: ('float: Epsilon Decay_rate'),
                 gamma: ('float: Discount Factor'),
                 n_step: ('int: The number of considered transitions in n-step TD'),
                 current_update_freq: ('int: The frequency of updating current Q-Network. 1 update per 4 (S,A,R,S) transitions in Double DQN paper'),
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
                 is_render: ('bool : Rendering option')=False,
                 is_test: ('boolean: To test')=False,
                 model_save_path: ('str: Model saving path')='./',
                 trained_model_path: ('str: Trained model path')='./'):

        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_save_path
        self.scores = [-10000]
        self.avg_scores = [-10000]

        self.env = env
        self.input_frames = input_frame
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.skipped_frame = skipped_frame
        self.epsilon = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.current_update_freq = current_update_freq
        self.target_update_freq = target_update_freq
        self.update_cnt = 0
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option
        self.is_render = is_render
        if is_render==True:
            self.vis = visdom.Visdom()
            self.state_plot = self.vis.image(np.zeros((input_dim, input_dim)))

        self.q_behave = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        self.q_target = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        if is_test:
            self.q_behave.load_state_dict(torch.load(trained_model_path))
            self.q_behave.eval()
            print("load completed.")

        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 

        self.memory = ReplayBuffer(self.buffer_size, (self.input_frames, self.input_dim, self.input_dim), self.batch_size)

        # Variables for multi-step TD
        self.n_step = n_step
        self.n_step_state_buffer = deque(maxlen=n_step)
        self.n_step_action_buffer = deque(maxlen=n_step)
        self.n_step_reward_buffer = deque(maxlen=n_step)
        self.gamma_list = [gamma**i for i in range(n_step)]

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        
        if np.random.random() < self.epsilon:
            return np.zeros(self.action_dim), self.env.action_space.sample()
        else:
            # if normalization is applied to the image such as devision by 255, MUST be expressed 'state/255' below.
            Qs = self.q_behave(torch.FloatTensor(state).to(self.device).unsqueeze(0)/255)
            return Qs.detach().cpu().numpy()[0], Qs.argmax().detach().item()

    def select_action_eval(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        ''' Evaluate with greedy policy '''
        # if normalization is applied to the image such as devision by 255, MUST be expressed 'state/255' below.
        Qs = self.q_behave(torch.FloatTensor(state).to(self.device).unsqueeze(0)/255)
        return Qs.detach().cpu().numpy()[0], Qs.argmax().detach().item()

    def processing_resize_and_gray(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Pure
        # frame = cv2.cvtColor(frame[:177, 32:128, :], cv2.COLOR_RGB2GRAY) # Boxing
        # frame = cv2.cvtColor(frame[2:198, 7:-7, :], cv2.COLOR_RGB2GRAY) # Breakout
        frame = cv2.resize(frame, dsize=(self.input_dim, self.input_dim)).reshape(self.input_dim, self.input_dim).astype(np.uint8)
        return frame 

    def get_init_state(self): 
        
        init_state = np.zeros((self.input_frames, self.input_dim, self.input_dim))
        init_frame = self.env.reset()
        init_state[0] = self.processing_resize_and_gray(init_frame)
        
        for i in range(1, self.input_frames): 
            action = self.env.action_space.sample()
            for j in range(self.skipped_frame):
                state, _, _, _ = self.env.step(action) 
            state, _, _, _ = self.env.step(action) 
            init_state[i] = self.processing_resize_and_gray(state) 
        return init_state

    def get_state(self, state, action):
        '''
        num_frames: how many frames to be merged
        input_size: hight and width of input resized image
        '''
        next_state = np.zeros((self.input_frames, self.input_dim, self.input_dim))
        for i in range(len(state)-1):
            next_state[i] = state[i+1]

        rewards = 0
        dones = 0
        for _ in range(self.skipped_frame):
            state, reward, done, _ = self.env.step(action) 
            rewards += reward 
            dones += int(done) 
        state, reward, done, _ = self.env.step(action) 
        next_state[-1] = self.processing_resize_and_gray(state) 
        rewards += reward 
        dones += int(done) 
        return rewards, next_state, dones
    
    def n_step_store(self, state, action, reward):
        '''method for n-step_TD'''
        self.n_step_state_buffer.append(state)
        self.n_step_action_buffer.append(action)
        self.n_step_reward_buffer.append(reward)

    def get_first_transitions(self, n_step=1):
        '''method for n-step_TD'''
        state = self.get_init_state()
        for _ in range(n_step):
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            self.n_step_store(state, action, reward)
            state = next_state
        rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
        self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
        return next_state

    def update_current_q_net(self):
        batch = self.memory.batch_load()
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def target_soft_update(self):
        for target_param, current_param in zip(self.q_target.parameters(), self.q_behave.parameters()):
            target_param.data.copy_(self.tau*current_param.data + (1.0-self.tau)*target_param.data)

    def target_hard_update(self):
        self.update_cnt = (self.update_cnt+1) % self.target_update_freq
        if self.update_cnt==0:
            self.q_target.load_state_dict(self.q_behave.state_dict())

    def train(self):
        tic = time.time()
        losses = []
        score = 0

        print("Storing initial buffer..")
        state = self.get_first_transitions(self.n_step)
        init_store_cnt = 0 # For restoring multi-step TD transitions until the agent reaches done state.
        while 1:
            init_store_cnt += 1
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            self.n_step_store(state, action, np.clip(reward, -1, 1))
            rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
            state = next_state
            if done: 
               state = self.get_first_transitions(self.n_step)
               if init_store_cnt>self.update_start: break

        print("Done. Start learning..")
        epi_Qs = []
        epsilons = [self.epsilon]
        history_store = []
        behave_update_cnt = 0
        for frame_idx in range(1, self.num_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action) 
            self.n_step_store(state, action, np.clip(reward, -1, 1)) 
            rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
            history_store.append([self.n_step_state_buffer[0], Qs, self.n_step_action_buffer[0], reward, next_state, done])

            behave_update_cnt = (behave_update_cnt+1) % self.current_update_freq
            if behave_update_cnt == 0:
                loss = self.update_current_q_net()
                losses.append(loss)

                if self.update_type=='hard':   self.target_hard_update()
                elif self.update_type=='soft': self.target_soft_update()
            score += reward
            epi_Qs.append(Qs)
            self.env_render(self.is_render, state, action, next_state, Qs)

            if done:
                self.model_save(frame_idx, score, self.scores, self.avg_scores, history_store, tic)
                self.plot_status(self.plot_option, frame_idx, score, self.scores, epi_Qs, losses, epsilons)
                score = 0        
                epi_Qs = []
                history_store = []
                state = self.get_first_transitions(self.n_step)
                epsilons.append(self.epsilon)
            else: state = next_state

            self._epsilon_step()

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))

    def model_save(self, frame_idx, score, scores, avg_scores, history_store, tic):
        '''model save when condition is satisfied'''
        if score > max(self.scores):
            torch.save(self.q_behave.state_dict(), self.model_path+'{}_Highest_Score_{}.pt'.format(frame_idx, score))
            training_time = round((time.time()-tic)/3600, 1)
            np.save(self.model_path+'{}_history_Highest_Score_{}_{}hrs.npy'.format(frame_idx, score, training_time), np.array(history_store, dtype=object))
            print("                    | Model saved. Highest score: {}, Training time: {}hrs".format(score, training_time), ' /'.join(os.getcwd().split('/')[-3:]))
        self.scores.append(score)
        if np.mean(self.scores[-10:]) > max(self.avg_scores):
            torch.save(self.q_behave.state_dict(), self.model_path+'{}_Avg_Score_{}.pt'.format(frame_idx, np.mean(self.scores[-10:])))
            training_time = round((time.time()-tic)/3600, 1)
            np.save(self.model_path+'{}_history_Score_{}_{}hrs.npy'.format(frame_idx, score, training_time), np.array(history_store, dtype=object))
            print("                    | Model saved. Recent scores: {}, Training time: {}hrs".format(self.scores[-10:], training_time), ' /'.join(os.getcwd().split('/')[-3:]))
        self.avg_scores.append(np.mean(self.scores[-10:]))

    def plot_status(self, is_plot, frame_idx, score, scores, epi_Qs, losses, epsilons):
        if is_plot=='inline': 
            self._plot_inline(frame_idx, scores, losses, epsilons)
        elif is_plot=='wandb': 
            wandb.log({'Score': score, 
                       'Number of frames': frame_idx,
                       'loss(10 frames avg)': np.mean(losses[-10:]), 
                       'epsilon': self.epsilon,
                       'Q(max_avg)': np.array(epi_Qs).max(1).mean(),
                       'Q(min_avg)': np.array(epi_Qs).min(1).mean(),
                       'Q(max)': np.array(epi_Qs).flatten().max(),
                       'Q(min)': np.array(epi_Qs).flatten().min()})
            print(score, end='\r')            
        else: 
            print(score, end='\r')

    def test(self):
        ''' Return an episodic history when the episode is done. '''
        num_of_no_op = np.random.randint(50)
        self.env.reset()
        for _ in range(num_of_no_op):
            self.env.step(0)
        episode_history = []
        state = self.get_first_transitions(self.n_step)
        while 1:
            Qs, action = self.select_action_eval(state)
            reward, next_state, done = self.get_state(state, action)
            self.env_render(self.is_render, state, action, next_state, Qs)
            episode_history.append((state, action, reward, next_state))
            state = next_state
            if done: break

        return episode_history

    def env_render(self, is_render, state, action, next_state, Qs):
        if is_render:
            # state plotting
            self.vis.image(
                state[0],
                opts={'title': f'Action taken at current state: {action}!', 
                      'caption': f'Q_value: {Qs.flatten()}\nArgmax: {Qs.argmax()}'},
                win=self.state_plot)
        else:pass

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
        # The update is conducted as in DoubleDQN.
        next_q = self.q_target(next_states).gather(1, self.q_behave(next_states).argmax(axis=1, keepdim=True)).detach()
        mask = 1 - dones
        target = (rewards + (mask * (self.gamma**self.n_step) * next_q)).to(self.device)

        # Gradient clipping is applied.
        loss = F.smooth_l1_loss(current_q, target)
        return loss

    def _plot_inline(self, frame_idx, scores, losses, epsilons):
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
