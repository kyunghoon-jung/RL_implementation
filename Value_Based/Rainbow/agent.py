import gym
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output

# Adabelief optimizer implementation
from adabelief_pytorch import AdaBelief

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.nn.utils import clip_grad_norm_
from torchsummary import summary

from qnetwork import QNetwork 
from replay_buffer import PrioritizedReplayBuffer

import wandb
import visdom

class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_frame: ('int: the number of channels of input image'),
                 input_dim: ('int: the width and height of pre-processed input image'),
                 initial_std: ('float: the standard deviation of noise'),  # Variables for NoisyNet
                 tot_train_frames: ('int: Total number of frames to be trained'),
                 skipped_frame: ('int: The number of skipped frames'),
                 gamma: ('float: Discount Factor'),
                 alpha: ('float: PER hyperparameter'), # Increasing alpha means emphasizing the priority 
                 beta: ('float: PER hyperparameter'),  # Increasing beta means more strong correction.
                 epsilon_for_priority: ('float: PER hyperparameter'),  # Preventing too small priority.
                 n_step: ('int: n-step size for Multi-step TD learning'),  # Variables for multi-step TD
                 Vmax: ('int: The maximum Q value')=10,    # Variables for Categprocal
                 Vmin: ('int: The minimum Q value')=-10,   # Variables for Categprocal
                 n_atoms: ('int: The number of atoms')=51, # Variables for Categprocal 
                 target_update_freq: ('int: Target Update Frequency (unit: frames)')=8000, # Defalut value is choosen as in the paper. (skipped frames=4)
                 behave_update_freq: ('int: Behavioral Network Update Frequency (unit: frames')=4,
                 grad_clip: ('float: The value bounding gradient norm.')=10,
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
                 trained_model_path: ('str: Trained model path')='',
                 is_render: ('bool: Whether rendering is on or off')=False,
                 vis_env_id: ('str: The environment name of visdom.')='main',
                 ): 
        
        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.scores = [-10000]
        self.avg_scores = [-10000]

        self.env = env
        self.input_frames = input_frame
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
        self.is_render = is_render
        self.initial_std = initial_std    # NoisyNet Variable
        self.epi_max_frame = 108000 # same as the Rainbow paper
        self.grad_clip = grad_clip

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

        self.q_behave = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim, initial_std, n_atoms=self.n_atoms).to(self.device)
        self.q_target = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim, initial_std, n_atoms=self.n_atoms).to(self.device)
        # Load trained model
        if trained_model_path:
            self.q_behave.load_state_dict(torch.load(trained_model_path))
            print("Trained model is loaded successfully.")
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        # self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate, eps=1e-4) # Epsilon value in the Rainbow paper 
        self.optimizer = AdaBelief(self.q_behave.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)

        # PER replay buffer
        self.memory = PrioritizedReplayBuffer(self.buffer_size, (self.input_frames, self.input_dim, self.input_dim), self.batch_size, self.alpha)

        # Variables for multi-step TD
        self.n_step = n_step
        self.n_step_state_buffer = deque(maxlen=n_step)
        self.n_step_action_buffer = deque(maxlen=n_step)
        self.n_step_reward_buffer = deque(maxlen=n_step)
        self.gamma_list = [gamma**i for i in range(n_step)]

        if self.plot_option == 'wandb':
            wandb.watch(self.q_behave)

        if self.is_render:
            print("Rendering mode is on.")
            self.action_list = {i:name for i, name in enumerate(env.unwrapped.get_action_meanings())}
            self.vis = visdom.Visdom(env=vis_env_id)
            self.vis.close()
            self.state_plots_dic = {}
            # Set plotting frames for each time step. (the number of time steps == n_stacked)
            self.state_plot = self.vis.heatmap(np.zeros((self.input_dim, self.input_dim)))
            self.state_meta = self.vis.text("Q_values.")

    def processing_resize_and_gray(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Pure
        # frame = cv2.cvtColor(frame[:177, 32:128, :], cv2.COLOR_RGB2GRAY) # Boxing
        # frame = cv2.cvtColor(frame[2:198, 7:-7, :], cv2.COLOR_RGB2GRAY) # Breakout
        frame = cv2.resize(frame, dsize=(self.input_dim, self.input_dim)).reshape(self.input_dim, self.input_dim).astype(np.uint8)
        return frame 

    def get_init_state(self, is_eval=False):

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

    def get_state(self, state, action, skipped_frame=0):
        '''
        input_frames: how many frames to be merged
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

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)/255
        
            # Categorical RL
            Expected_Qs = (self.q_behave(state)*self.expanded_support[0]).sum(2)
            action = Expected_Qs.argmax(1)
        return Expected_Qs.detach().cpu().numpy(), action.detach().item()

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
        clip_grad_norm_(self.q_behave.parameters(), self.grad_clip) # Gradient clipping mentioned in Dueling paper.
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
        epi_frame_count = 0
        while 1:
            init_store_cnt += 1
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            self.n_step_store(state, action, np.clip(reward, -1, 1))
            rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
            state = next_state
            epi_frame_count += 1
            if done or (epi_frame_count==self.epi_max_frame): 
               state = self.get_first_transitions(self.n_step)
               epi_frame_count = 0
               if init_store_cnt>self.update_start: break

        print("Done. Start learning..")
        epi_Qs=[]
        history_store = []
        behave_update_cnt = 0
        target_update_cnt = 0
        for frame_idx in range(1, self.tot_train_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            self.n_step_store(state, action, np.clip(reward, -1, 1)) 
            rewards = sum([gamma*reward for gamma, reward in zip(self.gamma_list, self.n_step_reward_buffer)])
            self.memory.store(self.n_step_state_buffer[0], self.n_step_action_buffer[0], rewards, next_state, done)
            history_store.append([self.n_step_state_buffer[0][0], Qs, self.n_step_action_buffer[0], reward, next_state[0], done])
            # print(f"(F:{frame_idx}, R:{rewards}, A:{action}), Qs: {np.round(Qs, 2)}", end='\r')
            
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
            epi_Qs.append(Qs)
            epi_frame_count += 1
            self.beta = min(self.beta+self.beta_step, 1.0) # for PER. beta is increased linearly up to 1.0 until the end of training
            if done or (epi_frame_count==self.epi_max_frame): # As in the paper, check if reaching the maximum steps per episode.
                self.model_save(frame_idx, score, self.scores, self.avg_scores, history_store, tic)
                self.plot_status(frame_idx, score, epi_Qs, losses)
                state = self.get_first_transitions(self.n_step)
                score = 0
                losses = []
                epi_Qs = []
                history_store = []
                epi_frame_count = 0
            else: state = next_state
            if self.is_render and (frame_idx % 5 == 0):
                self.render(state[0], action, Qs, frame_idx)
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

    def plot_status(self, frame_idx, score, epi_Qs, losses):
        if self.plot_option=='inline': 
            self._plot(frame_idx, self.scores, losses)
        elif self.plot_option=='wandb': 
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

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        # If normalization is used, it must be applied to 'state' and 'next_state' here. ex) state/255
        states = torch.FloatTensor(batch['states']).to(self.device) / 255
        next_states = torch.FloatTensor(batch['next_states']).to(self.device) / 255
        actions = torch.LongTensor(batch['actions']).to(self.device)
        weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)
        
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

    def test(self):
        ''' Return an episodic history when the episode is done. '''
        episode_history = []
        epi_frame_count = 0
        state = self.get_first_transitions(self.n_step)
        while 1:
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action)
            state = next_state
            epi_frame_count += 1
            episode_history.append(state[0], reward, next_state[0], done)
            if done or (epi_frame_count==self.epi_max_frame): 
               break
        return episode_history

    def render(self, state, action, Qs, frame_idx):
        self.vis.text(f'Frame:{frame_idx}, Q_current_value: {np.round(Qs, 3)}<br>Argmax: {Qs.argmax()}<br>', win=self.state_meta)
        self.vis.heatmap(
            state[::-1,:],
            opts={'title': f'Action: {self.action_list[action]}'},
            win=self.state_plot
        )

    def _plot(self, frame_idx, scores, losses):
        clear_output(True) 
        plt.figure(figsize=(20, 5), facecolor='w') 
        plt.subplot(121)  
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores) 
        plt.subplot(122) 
        plt.title('loss') 
        plt.plot(losses) 
        plt.show() 
