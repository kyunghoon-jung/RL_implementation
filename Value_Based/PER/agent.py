''' PER Agent '''

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
from adabelief_pytorch import AdaBelief

from qnetwork import QNetwork 
from replay_buffer import PrioritizedReplayBuffer

import wandb

class Agent:
    '''
    PER 구현 Agent
    변수 추가된 것 : alpha, beta, epsilon_for_priority
    alpha는 고정된 값을 유지하지만, beta는 1.0까지 점점 커지게 된다. (논문에서는 linearly decay up tp 1.0)
    alpha와 beta 둘 다 1.0으로 될수록, sampling할 때 priority를 중요하게 보는 것.
    epsilon_for_priority는 priority 값이 너무 작은 경우를 위해서 넣은 것.
    '''
    def __init__(self, 
                 env: 'Environment',
                 input_frame: ('int: the number of channels of input image'),
                 input_dim: ('int: the width and height of pre-processed input image'),
                 num_frames: ('int: Total number of frames'),
                 skipped_frame: ('int: The number of skipped frames'),
                 eps_decay: ('float: Epsilon Decay_rate'),
                 gamma: ('float: Discount Factor'),
                 target_update_freq: ('int: Target Update Frequency (by frames)'),
                 update_type: ('str: Update type for target network. Hard or Soft')='hard',
                 soft_update_tau: ('float: Soft update ratio')=None,
                 batch_size: ('int: Update batch size')=32,
                 buffer_size: ('int: Replay buffer size')=1000000,
                 alpha: ('int: Hyperparameter for how large prioritization is applied')=0.2,
                 beta: ('int: Hyperparameter for the annealing factor of importance sampling')=0.6,
                 epsilon_for_priority: ('float: Hyperparameter for adding small increment to the priority')=1e-8, 
                 current_update_freq: ('int: The frequency of updating current Q-Network. 1 update per 4 (S,A,R,S) transitions in Double DQN paper')=4,
                 update_start_buffer_size: ('int: Update starting buffer size')=50000,
                 learning_rate: ('float: Learning rate')=0.0004,
                 eps_min: ('float: Epsilon Min')=0.1,
                 eps_max: ('float: Epsilon Max')=1.0,
                 device_num: ('int: GPU device number')=0,
                 rand_seed: ('int: Random seed')=None,
                 plot_option: ('str: Plotting option')=False,
                 model_path: ('str: Model saving path')='./',
                 trained_model_path: ('str: Trained Model Path')=''):
        
        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        self.env = env
        self.input_frames = input_frame
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.skipped_frame = skipped_frame
        self.epsilon = eps_max
        self.epsilons = [] # For a case of plotting inline
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.current_update_freq = current_update_freq
        self.target_update_freq = target_update_freq // current_update_freq # Note: divided by current_update_freq.
        self.update_cnt = 0
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option
        self.scores = [-1000000]
        self.avg_scores = [-1000000]

        # hyperparameters for PER
        self.alpha = alpha
        self.beta = beta
        self.beta_step = (1.0 - beta) / num_frames
        self.epsilon_for_priority = epsilon_for_priority
        
        self.q_behave = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        self.q_target = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim).to(self.device)
        if trained_model_path:
            self.q_behave.load_state_dict(torch.load(trained_model_path))
            print("Pre-trained model is loaded successfully.")
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        # self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 
        self.optimizer = AdaBelief(self.q_behave.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)

        self.memory = PrioritizedReplayBuffer(self.buffer_size, (self.input_frames, self.input_dim, self.input_dim), self.batch_size, self.alpha)

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):

        if np.random.random() < self.epsilon:
            return np.zeros(self.action_dim), self.env.action_space.sample()
        else:
            # if normalization is applied to the image such as devision by 255, MUST be expressed 'state/255' below.
            Qs = self.q_behave(torch.FloatTensor(state).to(self.device).unsqueeze(0)/255)
            return Qs.detach().cpu().numpy()[0], Qs.argmax().detach().item()

    def processing_resize_and_gray(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Pure
        # frame = cv2.cvtColor(frame[:177, 32:128, :], cv2.COLOR_RGB2GRAY) # Boxing
        # frame = cv2.cvtColor(frame[2:198, 7:-7, :], cv2.COLOR_RGB2GRAY) # Breakout
        frame = cv2.resize(frame, dsize=(self.input_dim, self.input_dim)).reshape(self.input_dim, self.input_dim).astype(np.uint8)
        return frame 

    def get_state(self, state, action, skipped_frame=0):
        '''
        num_frames: how many frames to be merged
        input_size: hight and width of input resized image
        skipped_frame: how many frames to be skipped
        '''
        next_state = np.zeros((self.input_frames, self.input_dim, self.input_dim))
        for i in range(len(state)-1):
            next_state[i] = state[i+1]

        rewards = 0
        dones = 0
        for _ in range(skipped_frame):
            state, reward, done, _ = self.env.step(action) 
            rewards += reward 
            dones += int(done) 
        state, reward, done, _ = self.env.step(action) 
        next_state[-1] = self.processing_resize_and_gray(state) 
        rewards += reward 
        dones += int(done) 
        return rewards, next_state, dones

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

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
    
    def update_current_q_net(self):
        '''The diffent part between Dueling and PER in the Agent class
        : sample 당 weight가 곱해진다. 그래서 loss를 구할 때, sample-wise하도록 batch dimension을 유지한채로 loss를 구한다.
        '''
        batch = self.memory.batch_load(self.beta)
        weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(self.device)
        sample_wise_loss = self._compute_loss(batch) # PER: shape of loss -> (batch, 1) 
        loss = (weights * sample_wise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # For PER: update priorities of the samples. 
        sample_wise_loss = sample_wise_loss.detach().cpu().numpy()
        batch_priorities = sample_wise_loss + self.epsilon_for_priority
        # sampling 된 batch에 대한 priority를 update하는 부분이다. batch_priorities는 batch dimension이 살아있어서 
        # update_priorities 함수를 정의할 때 유념해야 한다. 여기서는 이 함수에서 그대로 받은 후, .flatten()을 해주었다.
        self.memory.update_priorities(batch['indices'], batch_priorities)

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
        state = self.get_init_state()
        for frame_idx in range(1, self.update_start+1):
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)  # Store a reward clipped btw -1 ~ 1
            state = next_state
            if done: state = self.get_init_state()

        print("Done. Start learning..")
        epi_Qs = []
        history_store = []
        current_update_count = 0
        for frame_idx in range(1, self.num_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            # print("(R:{}, A:{}), Qs: {}".format(reward, action, np.round(Qs, 3)), end='\r')
            self.store(state, action, np.clip(reward, -1, 1), next_state, done) # Store a reward clipped btw -1 ~ 1
            history_store.append([state, Qs, action, reward, next_state, done])

            current_update_count = (current_update_count+1) % self.current_update_freq
            if current_update_count == 0:
                loss = self.update_current_q_net()
                losses.append(loss)

                if self.update_type=='hard':   self.target_hard_update()
                elif self.update_type=='soft': self.target_soft_update()

            epi_Qs.append(Qs)
            score += reward
            if done:
                self.model_save(frame_idx, score, history_store, tic)
                self.plot_status(self.plot_option, frame_idx, score, losses, epi_Qs)
                state = self.get_init_state()
                history_store = []
                score = 0
                epi_Qs = []
            else: state = next_state

            self._epsilon_step()

            # for PER. beta is increased linearly up to 1.0
            self.beta = min(self.beta+self.beta_step, 1.0)

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))

    def model_save(self, frame_idx, score, history_store, tic):
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

    def plot_status(self, is_plot, frame_idx, score, losses, epi_Qs):
        if is_plot=='inline': 
            self._plot(frame_idx, self.scores, losses, self.epsilons)
        elif is_plot=='wandb': 
            wandb.log({'Score': score, 
                       'Episodic accumulated frames': len(epi_Qs), 
                       'Episode Count': len(self.scores)-1, 
                       'loss(10 frames avg)': np.mean(losses[-10:]),
                       'Q(max_avg)': np.array(epi_Qs).max(1).mean(),
                       'Q(min_avg)': np.array(epi_Qs).min(1).mean(),
                       'Q(mean)': np.array(epi_Qs).flatten().mean(),
                       'Q(max)': np.array(epi_Qs).flatten().max(),
                       'Q(min)': np.array(epi_Qs).flatten().min()}, step=frame_idx)
            print(score, end='\r')
        else: 
            print(score, end='\r')

    def _epsilon_step(self):
        ''' Epsilon decay control '''
        eps_decay_list = [self.eps_decay,self.eps_decay/2.5,self.eps_decay/3.5,self.eps_decay/5.5] 
        self.epsilons.append(self.epsilon)
        if self.epsilon>0.30:
            self.epsilon = max(self.epsilon-eps_decay_list[0], 0.1)
        elif self.epsilon>0.27:
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

        next_q = self.q_target(next_states).gather(1, self.q_behave(next_states).argmax(axis=1, keepdim=True)).detach()
        mask = 1 - dones
        target = (rewards + (mask * self.gamma * next_q)).to(self.device)

        # For PER, the shape of loss should be (batch, 1). Therefore, here is using "reduction='none'" option.
        # Gradient clipping [-1, 1] is applied(=F.smooth_l1_loss). 
        loss = F.smooth_l1_loss(current_q, target, reduction="none")
        return loss

    def _plot(self, frame_idx, scores, losses, epsilons):
        ''' For executing in Jupyter notebook '''
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