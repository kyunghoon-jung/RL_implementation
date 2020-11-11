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

from qnetwork import QNetwork, QNetwork_1dim
from replay_buffer import PrioritizedReplayBuffer

import wandb

class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_frame: ('int: the number of channels of input image'),
                 input_dim: ('int: the width and height of pre-processed input image'),
                 input_type: ('str: the type of input dimension'),
                 num_frames: ('int: Total number of frames'),
                 skipped_frame: ('int: The number of skipped frames'),
                 eps_decay: ('float: Epsilon Decay_rate'),
                 gamma: ('float: Discount Factor'),
                 target_update_freq: ('int: Target Update Frequency (by frames)'),
                 update_type: ('str: Update type for target network. Hard or Soft')='hard',
                 soft_update_tau: ('float: Soft update ratio')=None,
                 batch_size: ('int: Update batch size')=32,
                 buffer_size: ('int: Replay buffer size')=1000000,
                 alpha: ('int: Hyperparameter for how large prioritization is applied')=0.5,
                 beta: ('int: Hyperparameter for the annealing factor of importance sampling')=0.5,
                 epsilon_for_priority: ('float: Hyperparameter for adding small increment to the priority')=1e-6, 
                 update_start_buffer_size: ('int: Update starting buffer size')=50000,
                 learning_rate: ('float: Learning rate')=0.0004,
                 eps_min: ('float: Epsilon Min')=0.1,
                 eps_max: ('float: Epsilon Max')=1.0,
                 device_num: ('int: GPU device number')=0,
                 rand_seed: ('int: Random seed')=None,
                 plot_option: ('str: Plotting option')=False,
                 model_path: ('str: Model saving path')='./'):

        try:
            self.action_dim = env.action_space.n
        except AttributeError:
            self.action_dim = env.action_space.shape[0]
            
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
        
        # hyper parameters for PER
        self.alpha = alpha
        self.beta = beta
        self.beta_step = (1.0 - beta) / num_frames
        self.epsilon_for_priority = epsilon_for_priority
        
        if input_type=='1-dim':
            self.q_current = QNetwork_1dim(self.input_dim, self.action_dim).to(self.device)
            self.q_target = QNetwork_1dim(self.input_dim, self.action_dim).to(self.device)
        else:
            self.q_current = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
            self.q_target = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_current.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_current.parameters(), lr=learning_rate) 

        if input_type=='1-dim':
            self.memory = PrioritizedReplayBuffer(self.buffer_size, self.input_dim, self.batch_size, self.alpha, input_type)
        else:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, (self.input_frames, self.input_dim, self.input_dim), self.batch_size, self.alpha, input_type)

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        
        if np.random.random() < self.epsilon:
            return np.zeros(self.action_dim), self.env.action_space.sample()
        else:
            Qs = self.q_current(torch.FloatTensor(state).to(self.device))
            return Qs.detach().cpu().numpy(), Qs.argmax().detach().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def update_current_q_net(self):
        '''The diffent method between Dueling and PER in the Agent class'''
        batch = self.memory.batch_load(self.beta)
        weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(self.device)
        sample_wise_loss = self._compute_loss(batch) # PER: shape of loss -> (batch, 1) 
        loss = torch.mean(sample_wise_loss*weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # For PER: update priorities of the samples. 
        sample_wise_loss = sample_wise_loss.detach().cpu().numpy()
        batch_priorities = sample_wise_loss + self.epsilon_for_priority
        self.memory.update_priorities(batch['indices'], batch_priorities)

        return loss.item()

    def target_soft_update(self):
        for target_param, current_param in zip(self.q_target.parameters(), self.q_current.parameters()):
            target_param.data.copy_(self.tau*current_param.data + (1.0-self.tau)*target_param.data)

    def target_hard_update(self):
        self.update_cnt = (self.update_cnt+1) % self.target_update_freq
        if self.update_cnt==0:
            self.q_target.load_state_dict(self.q_current.state_dict())

    def train(self):
        tic = time.time()
        losses = []
        scores = []
        epsilons = []
        avg_scores = [[-1000]]

        score = 0

        print("Storing initial buffer..")
        state = self.env.reset()
        for frame_idx in range(1, self.update_start+1):
            _, action = self.select_action(state) 
            next_state, reward, done, _ = self.env.step(action) 
            self.store(state, action, reward, next_state, done)
            state = next_state
            if done: state = self.env.reset()

        print("Done. Start learning..")
        history_store = []
        for frame_idx in range(1, self.num_frames+1):
            Qs, action = self.select_action(state) 
            next_state, reward, done, _ = self.env.step(action) 
            self.store(state, action, reward, next_state, done)
            history_store.append([state, Qs, action, reward, next_state, done])
            loss = self.update_current_q_net()

            if self.update_type=='hard':   self.target_hard_update()
            elif self.update_type=='soft': self.target_soft_update()

            score += reward
            losses.append(loss)

            if done:
                scores.append(score)
                if np.mean(scores[-10:]) > max(avg_scores):
                    torch.save(self.q_current.state_dict(), self.model_path+'{}_Score:{}.pt'.format(frame_idx, np.mean(scores[-10:])))
                    training_time = round((time.time()-tic)/3600, 1)
                    np.save(self.model_path+'{}_history_Score_{}_{}hrs.npy'.format(frame_idx, score, training_time), np.array(history_store))
                    print("          | Model saved. Recent scores: {}, Training time: {}hrs".format(scores[-10:], training_time), ' /'.join(os.getcwd().split('/')[-3:]))
                avg_scores.append(np.mean(scores[-10:]))

                if self.plot_option=='inline': 
                    scores.append(score)
                    epsilons.append(self.epsilon)
                    self._plot(frame_idx, scores, losses, epsilons)
                elif self.plot_option=='wandb': 
                    Q_avg_actions = np.array(history_store)[:,1].mean().mean().item()
                    wandb.log({'Score': score,
                               'loss(10 frames avg)': np.mean(losses[-10:]), 
                               'Q(mean)': Q_avg_actions, 
                               'Epsilon': self.epsilon, 
                               'beta': self.beta})
                    print(score, end='\r')
                else: 
                    print(self.epsilon, score, end='\r')

                score=0
                state = self.env.reset()
                # if frame_idx>10000:
                #     np.save('./history_Cart.npy', np.array(history_store))
                #     print("History saved.")
            else: state = next_state

            self._epsilon_step()

            # self.beta = min(self.beta+self.beta_step, 1.0) # for PER. beta is increased linearly up to 1.0
            fraction = min(frame_idx / self.num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))

    def _epsilon_step(self):
        ''' Epsilon decay control '''
        eps_decay = [self.eps_decay, self.eps_decay/2.5, self.eps_decay/3.5, self.eps_decay/5.5] 

        if self.epsilon>0.30:
            self.epsilon = max(self.epsilon-eps_decay[0], 0.1)
        elif self.epsilon>0.27:
            self.epsilon = max(self.epsilon-eps_decay[1], 0.1)
        elif self.epsilon>1.7:
            self.epsilon = max(self.epsilon-eps_decay[2], 0.1)
        else:
            self.epsilon = max(self.epsilon-eps_decay[3], 0.1)

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        # If normalization is used, it must be applied to 'state' and 'next_state' here. ex) state/255
        states = torch.FloatTensor(batch['states']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        actions = torch.LongTensor(batch['actions'].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)

        current_q = self.q_current(states).gather(1, actions)
        # The next line is the only difference from Vanila DQN.
        next_q = self.q_target(next_states).gather(1, self.q_current(next_states).argmax(axis=1, keepdim=True)).detach()
        mask = 1 - dones
        target = (rewards + (mask * self.gamma * next_q)).to(self.device)

        # For PER, the shape of loss is (batch, 1). Therefore, using "reduction='none'" option.
        sample_wise_loss = F.smooth_l1_loss(current_q, target, reduction="none")
        return sample_wise_loss

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