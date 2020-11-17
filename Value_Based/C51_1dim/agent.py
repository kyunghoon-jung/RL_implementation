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

from qnetwork import QNetwork 
from replay_buffer import ReplayBuffer

import wandb
class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_dim: ('int: the width and height of pre-processed input image'),
                 num_frames: ('int: Total number of frames'),
                 eps_decay: ('float: Epsilon Decay_rate'),
                 gamma: ('float: Discount Factor'),
                 target_update_freq: ('int: Target Update Frequency (unit: frames)'),
                 current_update_freq: ('int: Behavioral Network Update Frequency (unit: frames'),
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
                 n_atoms: ('int: The number of atoms')=51, # Variables for Categprocal 
                 Vmax: ('int: The maximum Q value')=10,    # Variables for Categprocal
                 Vmin: ('int: The minimum Q value')=-10,   # Variables for Categprocal
                 ): 
        
        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        self.env = env
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.epsilon = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.current_update_freq = current_update_freq
        self.update_cnt = 0
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option
        self.scores = [-100000]
        self.avg_scores = [-100000]

        # Variables for C51
        self.n_atoms = n_atoms                            
        self.Vmin = Vmin                                  
        self.Vmax = Vmax                                  
        self.dz = (Vmax - Vmin) / (n_atoms - 1)           
        self.support = torch.linspace(Vmin, Vmax, n_atoms).to(self.device) 
        self.expanded_support = self.support.expand((batch_size, self.action_dim, n_atoms)).to(self.device)
        
        self.q_behave = QNetwork(self.input_dim, self.action_dim, n_atoms=self.n_atoms).to(self.device)
        self.q_target = QNetwork(self.input_dim, self.action_dim, n_atoms=self.n_atoms).to(self.device)
        # Load trained model
        # self.q_behave.load_state_dict(torch.load("/home/ubuntu/playground/MacaronRL_prev/Value_Based/DuelingDQN/model_save/890_BreakoutNoFrameskip-v4_num_f:10000000_eps_dec:8.3e-07f_gamma:0.99_tar_up_frq:150f_up_type:hard_soft_tau:0.002f_batch:32_buffer:750000f_up_start:50000_lr:0.0001f_eps_min:0.1_device:0_rand:None_0/2913941_Score:37.6.pt"))
        # print("load completed.")
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 

        self.memory = ReplayBuffer(self.buffer_size, self.input_dim, self.batch_size)

        if self.plot_option == 'wandb':
            wandb.watch(self.q_behave)

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        if np.random.random() < self.epsilon:
            self._epsilon_step()
            return np.zeros(self.action_dim), self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
                # Categorical RL
                Expected_Qs = (self.q_behave(state)*self.expanded_support[0]).sum(2)
                action = Expected_Qs.argmax(1)
            return Expected_Qs.detach().cpu().numpy(), action.detach().item()

    def processing_resize_and_gray(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Pure
        # frame = cv2.cvtColor(frame[:177, 32:128, :], cv2.COLOR_RGB2GRAY) # Boxing
        # frame = cv2.cvtColor(frame[2:198, 7:-7, :], cv2.COLOR_RGB2GRAY) # Breakout
        frame = cv2.resize(frame, dsize=(self.input_dim, self.input_dim)).reshape(self.input_dim, self.input_dim).astype(np.uint8)
        return frame 

    def get_init_state(self):

        init_state = self.env.reset()
        for _ in range(0): # Random initial starting point.
            action = self.env.action_space.sample()
            init_state, _, _, _ = self.env.step(action) 
        return init_state

    def get_state(self, state, action):

        next_state, reward, done, _ = self.env.step(action)
        return reward, next_state, done

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

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
        epsilons = []
        avg_scores = [-1000]
        score = 0

        print("Storing initial buffer..")
        state = self.get_init_state()
        for frame_idx in range(1, self.update_start+1):
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)
            state = next_state
            if done: state = self.get_init_state()

        print("Done. Start learning..")
        history_store = []
        behave_update_cnt = 0
        for frame_idx in range(1, self.num_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)
            history_store.append([state, Qs, action, reward, next_state, done])
            loss = self.update_current_q_net()

            behave_update_cnt = (behave_update_cnt+1)%self.current_update_freq
            if self.update_type=='soft': self.target_soft_update()
            elif behave_update_cnt == 0: self.target_hard_update()
            
            score += reward
            losses.append(loss)

            if done:
                self.model_save(frame_idx, score, self.scores, self.avg_scores, history_store, tic)
                epsilons.append(self.epsilon)
                self.plot_status(self.plot_option, frame_idx, score, self.scores, losses, epsilons)
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
            print("                    | Model saved. Recent scores: {}, Training time: {}hrs".format(np.round(self.scores[-10:],2), training_time), ' /'.join(os.getcwd().split('/')[-3:]))
        self.avg_scores.append(np.mean(self.scores[-10:]))

    def plot_status(self, is_plot, frame_idx, score, scores, losses, epsilons):
        if is_plot=='inline': 
            self._plot_inline(frame_idx, scores, losses, epsilons)
        elif is_plot=='wandb': 
            wandb.log({'Score': score, 'Number of frames': frame_idx, 'loss(10 frames avg)': np.mean(losses[-10:]), 'Epsilon': self.epsilon})
            print(score, end='\r')
        else: 
            print(score, end='\r')

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        # If normalization is used, it must be applied to 'state' and 'next_state' here. ex) state/255
        states = torch.FloatTensor(batch['states']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)
        
        log_behave_Q_dist = self.q_behave(states)[range(self.batch_size), actions].log()
        with torch.no_grad():
            # Compuating projected distribution for a categorical loss
            behave_next_Q_dist = self.q_behave(next_states)
            next_actions = torch.sum(behave_next_Q_dist*self.expanded_support, 2).argmax(1)
            target_next_Q_dist = self.q_target(next_states)[range(self.batch_size), next_actions] # Double DQN.
            Tz = rewards + self.gamma*(1 - dones)*self.expanded_support[:,0]
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

        loss = torch.sum(-proj_dist*log_behave_Q_dist, 1).mean()
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
