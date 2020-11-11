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
                 is_render: ('bool: Whether to render in real-time')=False,
                 is_test: ('bool: Whether to run in test mode')=False,
                 trained_model_path: ('bool: The pre-trained model path')=''
                 ): 
        
        self.action_dim = env.action_space.n
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        self.env = env
        self.input_frames = input_frame
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.skipped_frame = skipped_frame
        self.epsilon = eps_max
        self.epsilons = [] # For plotting inline in jupyter notebook.
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
        
        self.q_behave = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim, n_atoms=self.n_atoms).to(self.device)
        self.q_target = QNetwork((self.input_frames, self.input_dim, self.input_dim), self.action_dim, n_atoms=self.n_atoms).to(self.device)

        self.is_render = is_render
        if trained_model_path:
            self.q_behave.load_state_dict(torch.load(trained_model_path)) # load trained model. 
            print("Pre-trained Model loaded.")
            if is_test:
                self.q_behave.eval()
                print("*** Evaluation Mode ***")
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 

        self.memory = ReplayBuffer(self.buffer_size, (self.input_frames, self.input_dim, self.input_dim), self.batch_size)

        if self.plot_option == 'wandb':
            wandb.watch(self.q_behave)

        if is_render:
            eid = 'RL_Env'
            env_name = eid+str(np.random.randint(1000))
            self.is_render = is_render
            self.vis = visdom.Visdom(env=env_name)
            self.state_plots_dic = {}
            for i in range(self.input_frames):
                self.state_plots_dic[f'input_frame_{i}'] = self.vis.heatmap(np.zeros((input_dim, input_dim)))
            self.state_plots_dic['Status'] = self.vis.text("Status infos.")

    def processing_resize_and_gray(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Pure
        # frame = cv2.cvtColor(frame[:177, 32:128, :], cv2.COLOR_RGB2GRAY) # Boxing
        # frame = cv2.cvtColor(frame[2:198, 7:-7, :], cv2.COLOR_RGB2GRAY) # Breakout
        frame = cv2.resize(frame, dsize=(self.input_dim, self.input_dim)).reshape(self.input_dim, self.input_dim).astype(np.uint8)
        return frame 

    def select_action(self, state: 'Must be pre-processed in the same way while updating current Q network. See def _compute_loss'):
        
        if np.random.random() < self.epsilon:
            return np.zeros((self.action_dim, self.n_atoms)), np.zeros(self.action_dim), self.env.action_space.sample()
        else:
            # if normalization is applied to the image such as devision by 255, MUST be expressed 'state/255' below.
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)/255
                # Categorical RL
                Qs = self.q_behave(state)*self.expanded_support[0]
                Expected_Qs = Qs.sum(2)
                action = Expected_Qs.argmax(1)
            return Qs.detach().cpu().numpy()[0], Expected_Qs.detach().cpu().numpy()[0], action.detach().item()

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
        score = 0

        print("Storing initial buffer..")
        state = self.get_init_state()
        for frame_idx in range(1, self.update_start+1):
            _, _, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)
            state = next_state
            if done: state = self.get_init_state()

        print("Done. Start learning..")
        epi_Qs = []
        history_store = []
        behave_update_cnt = 0
        for frame_idx in range(1, self.num_frames+1):
            _, Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(state, action, skipped_frame=self.skipped_frame)
            self.store(state, action, np.clip(reward, -1, 1), next_state, done)
            history_store.append([state, Qs, action, reward, next_state, done])

            behave_update_cnt = (behave_update_cnt+1) % self.current_update_freq
            if behave_update_cnt == 0:
                loss = self.update_current_q_net()
                losses.append(loss)
                if self.update_type=='hard':   self.target_hard_update()
                elif self.update_type=='soft': self.target_soft_update()
            
            epi_Qs.append(Qs)
            score += reward

            if done:
                self.model_save(frame_idx, score, self.scores, self.avg_scores, history_store, tic)
                self.plot_status(self.plot_option, frame_idx, score, losses, epi_Qs)
                score = 0
                epi_Qs = []
                history_store = []
                state = self.get_init_state()
            else: state = next_state

            self._epsilon_step()

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))
        
    def test(self):
        ''' Return an episodic history when the episode is done. '''
        episode_history = []
        state = self.get_init_state()
        for frame_idx in range(1000000):
            time.sleep(0.5)
            Qs, Exp_Qs, action = self.select_action(state) 
            reward, next_state, done = self.get_state(state, action)
            episode_history.append([state, action, reward, Exp_Qs, Qs])
            self._render(state, action, Exp_Qs)
            state = next_state
            print(Qs.shape)
            print("distributional_Qs", np.round(Qs, 3))
            print("distributional_Qs_sum", np.round(Qs.sum(1), 3))
            print('Frame: ', frame_idx, "Expexted_Q_values: ", np.round(Exp_Qs, 3), "Action: ", action, "Reward: ", reward)
            if done: break
        return episode_history

    def _render(self, state, action, Qs):
        if self.is_render:
            self.vis.text(f'Q_current_value: {np.round(Qs, 3)}<br>Argmax: {Qs.argmax()}', win=self.state_plots_dic['Status'])
            for i in range(self.input_frames):
                self.vis.heatmap(
                    state[0], 
                    opts={'title': f'At time = {i}. Action: {action}'},
                    win=self.state_plots_dic[f'input_frame_{i}'])
        else:pass

    def _epsilon_step(self):
        ''' Epsilon decay control '''
        eps_decay_list = [self.eps_decay, self.eps_decay/2.5, self.eps_decay/3.5, self.eps_decay/5.5] 
        self.epsilons.append(self.epsilon)
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

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        # If normalization is used, it must be applied to 'state' and 'next_state' here. ex) state/255
        states = torch.FloatTensor(batch['states']).to(self.device) / 255
        next_states = torch.FloatTensor(batch['next_states']).to(self.device) / 255
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)
        
        log_behave_Q_dist = self.q_behave(states)[range(self.batch_size), actions].log()
        with torch.no_grad():
            # Computing projected distribution for a categorical loss
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
