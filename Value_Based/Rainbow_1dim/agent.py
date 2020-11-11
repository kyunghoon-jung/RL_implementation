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
from adabelief_pytorch import AdaBelief

from qnetwork import QNetwork 
from replay_buffer import PrioritizedReplayBuffer

import wandb
class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_dim: ('int: the width and height of pre-processed input image'),
                 initial_std: ('float: the standard deviation of noise'),  # Variables for NoisyNet
                 tot_train_frames: ('int: Total number of frames to be trained'),
                 skipped_frame: ('int: The number of skipped frames'),
                 gamma: ('float: Discount Factor'),
                 alpha: ('float: PER hyperparameter'), # Increasing alpha means emphasizing the priority 
                 beta: ('float: PER hyperparameter'),  # Increasing beta means more strong correction.
                 epsilon_for_priority: ('float: PER hyperparameter'),  # Preventing too small priority.
                 n_step: ('int: n-step size for Multi-step TD learning'),  # Variables for multi-step TD
                 target_update_freq: ('int: Target Update Frequency (unit: frames)'),
                 behave_update_freq: ('int: Behavioral Network Update Frequency (unit: frames'),
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
                 n_atoms: ('int: The number of atoms')=51, # Variables for Categprocal 
                 Vmax: ('int: The maximum Q value')=10,    # Variables for Categprocal
                 Vmin: ('int: The minimum Q value')=-10,   # Variables for Categprocal
                 ): 
        
        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.scores = [-10000]
        self.avg_scores = [-10000]

        self.env = env
        self.input_dim = input_dim
        self.tot_train_frames = tot_train_frames
        self.skipped_frame = skipped_frame
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.behave_update_freq = behave_update_freq
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option
        self.initial_std = initial_std    # NoisyNet Variable

        # hyper parameters and varibles for C51
        self.n_atoms = n_atoms                           
        self.Vmin = Vmin                                  
        self.Vmax = Vmax                                  
        self.dz = (Vmax - Vmin) / (n_atoms - 1)           
        self.support = torch.linspace(Vmin, Vmax, n_atoms).to(self.device) 
        self.expanded_support = self.support.expand((batch_size, self.action_dim, n_atoms)).to(self.device)

        # hyper parameters for PER
        self.alpha = alpha
        self.beta = beta
        self.beta_step = (1.0 - beta) / tot_train_frames
        self.epsilon_for_priority = epsilon_for_priority

        self.q_behave = QNetwork(self.input_dim, self.action_dim, initial_std, n_atoms=self.n_atoms).to(self.device)
        self.q_target = QNetwork(self.input_dim, self.action_dim, initial_std, n_atoms=self.n_atoms).to(self.device)
        # Load trained model
        # self.q_behave.load_state_dict(torch.load("/home/ubuntu/playground/MacaronRL_prev/Value_Based/DuelingDQN/model_save/890_BreakoutNoFrameskip-v4_num_f:10000000_eps_dec:8.3e-07f_gamma:0.99_tar_up_frq:150f_up_type:hard_soft_tau:0.002f_batch:32_buffer:750000f_up_start:50000_lr:0.0001f_eps_min:0.1_device:0_rand:None_0/2913941_Score:37.6.pt"))
        # print("load completed.")
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        # self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 
        self.optimizer = AdaBelief(self.q_behave.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)

        # PER replay buffer
        self.memory = PrioritizedReplayBuffer(self.buffer_size, self.input_dim, self.batch_size, self.alpha)

        # Variables for multi-step TD
        self.n_step = n_step
        self.n_step_state_buffer = deque(maxlen=n_step)
        self.n_step_action_buffer = deque(maxlen=n_step)
        self.n_step_reward_buffer = deque(maxlen=n_step)
        self.gamma_list = [gamma**i for i in range(n_step)]

        if self.plot_option == 'wandb':
            wandb.watch(self.q_behave)

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

            # Categorical RL
            Expected_Qs = (self.q_behave(state)*self.expanded_support[0]).sum(2)
            action = Expected_Qs.argmax(1)
        return Expected_Qs.detach().cpu().numpy(), action.detach().item()

    def get_init_state(self):

        init_state = self.env.reset()
        for _ in range(0): # Random initial starting point.
            action = self.env.action_space.sample()
            init_state, _, _, _ = self.env.step(action) 
        return init_state

    def get_state(self, state, action):

        next_state, reward, done, _ = self.env.step(action)
        return reward, next_state, done

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
        batch = self.memory.batch_load(self.beta)
        loss, sample_wise_loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # For PER: update priorities of the samples. 
        sample_wise_loss = sample_wise_loss.detach().cpu().numpy()
        batch_priorities = sample_wise_loss + self.epsilon_for_priority
        self.memory.update_priorities(batch['indices'], batch_priorities)
        return loss.item()

    def target_soft_update(self):
        for target_param, current_param in zip(self.q_target.parameters(), self.q_behave.parameters()):
            target_param.data.copy_(self.tau*current_param.data + (1.0-self.tau)*target_param.data)

    def target_hard_update(self):
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
            if done:
                reward = -1
                self.n_step_store(state, action, np.clip(reward, -1, 1)) 
                rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            else:
                self.n_step_store(state, action, np.clip(reward, -1, 1)) 
                rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
            state = next_state
            if done: 
               state = self.get_first_transitions(self.n_step)
               if init_store_cnt>self.update_start: break

        print("Done. Start learning..")
        history_store = []
        behave_update_cnt = 0
        target_update_cnt = 0
        for frame_idx in range(1, self.tot_train_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            if done:
                reward = -1
                self.n_step_store(state, action, np.clip(reward, -1, 1)) 
                rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            else:
                self.n_step_store(state, action, np.clip(reward, -1, 1)) 
                rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
            history_store.append([self.n_step_state_buffer[0], Qs, self.n_step_action_buffer[0], reward, next_state, done])
            
            behave_update_cnt = (behave_update_cnt+1) % self.behave_update_freq 
            target_update_cnt = (target_update_cnt+1) % self.target_update_freq
            if behave_update_cnt == 0:
                loss = self.update_current_q_net()
                losses.append(loss)
                if self.update_type=='soft': self.target_soft_update()
                elif target_update_cnt==0: self.target_hard_update()
                self.q_behave.init_noise()
                self.q_target.init_noise()
            score += reward
            self.beta = min(self.beta+self.beta_step, 1.0) # for PER. beta is increased linearly up to 1.0 until the end of training
            if done:
                self.model_save(frame_idx, score, self.scores, self.avg_scores, history_store, tic)
                self.plot_status(self.plot_option, frame_idx, score, self.scores, losses)
                state = self.get_first_transitions(self.n_step)
                history_store = []
                score=0
            else: state = next_state
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

    def plot_status(self, is_plot, frame_idx, score, scores, losses):
        if is_plot=='inline': 
            self._plot_inline(frame_idx, scores, losses)
        elif is_plot=='wandb': 
            wandb.log({'Score': score, 'Number of frames': frame_idx, 'loss(10 frames avg)': np.mean(losses[-10:])})
            print(score, end='\r')
        else: 
            print(f'Frame_idx:{frame_idx}, Score: {score}, Episode: {len(scores)-1}', end='\r')

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        states = torch.FloatTensor(batch['states']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)
        weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(self.device)
        
        log_behave_Q_dist = self.q_behave(states)[range(self.batch_size), actions].log()
        with torch.no_grad():
            # Compuating projected distribution for a categorical loss
            behave_next_Q_dist = self.q_behave(next_states)
            next_actions = torch.sum(behave_next_Q_dist*self.expanded_support, 2).argmax(1)
            target_next_Q_dist = self.q_target(next_states)[range(self.batch_size), next_actions] # Double DQN.

            Tz = rewards + (self.gamma**self.n_step)*(1 - dones)*self.expanded_support[:,0]
            Tz.clamp_(self.Vmin, self.Vmax)
            b = (Tz - self.Vmin) / self.dz
            l = b.floor().long()
            u = b.ceil().long()

            l[(l==u) & (u>0)] -= 1
            u[(u==0) & (l==0)] += 1

            batch_init_indices = torch.linspace(0, (self.batch_size-1)*self.n_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)

            proj_dist = torch.zeros(self.batch_size, self.n_atoms).to(self.device)
            proj_dist.view(-1).index_add_(0, (l+batch_init_indices).view(-1), (target_next_Q_dist*(u-b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u+batch_init_indices).view(-1), (target_next_Q_dist*(b-l)).view(-1))

        sample_wise_loss = torch.sum(-proj_dist*log_behave_Q_dist, 1)
        loss = (weights * sample_wise_loss).mean()

        return loss, sample_wise_loss

    def _plot_inline(self, frame_idx, scores, losses):
        clear_output(True) 
        plt.figure(figsize=(20, 5), facecolor='w') 
        plt.subplot(121)  
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores) 
        plt.subplot(122) 
        plt.title('loss') 
        plt.plot(losses) 
        plt.show() 
