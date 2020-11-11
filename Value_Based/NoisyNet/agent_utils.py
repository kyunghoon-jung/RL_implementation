import numpy as np
import time
import os
import cv2
import torch
import matplotlib.pyplot as plt
import wandb
from qnetwork import * 
from replay_buffer import ReplayBuffer
from IPython.display import clear_output

class Agent_Utils:

    def __init__(self, agent):
        
        self.losses = []
        if agent.is_noisy == True:
            print("NoisyNet is applied.")
            if agent.input_feature_type == '1dim':
                print("The input feature is 1-dimension.")
                self.select_action = self.select_action_1dim_noisynet
                self.q_current = Noisy_QLinearNetwork(agent.input_dim, 
                                                      agent.action_dim, 
                                                      agent.initial_std
                                                      ).to(agent.device)
                self.q_target = Noisy_QLinearNetwork(agent.input_dim, 
                                                     agent.action_dim, 
                                                     agent.initial_std
                                                     ).to(agent.device)
                self.memory = ReplayBuffer(agent.buffer_size, 
                                           agent.input_dim, 
                                           agent.batch_size, 
                                           agent.input_feature_type)
                
            elif agent.input_feature_type == '2dim':
                print("The input feature is 2-dimension.")
                self.select_action = self.select_action_2dim_noisynet
                self.q_current = Noisy_QConvNetwork((agent.input_frames, agent.input_dim, agent.input_dim), 
                                                     agent.action_dim, 
                                                     agent.initial_std
                                                     ).to(agent.device)
                self.q_target = Noisy_QConvNetwork((agent.input_frames, agent.input_dim, agent.input_dim), 
                                                     agent.action_dim, 
                                                     agent.initial_std
                                                     ).to(agent.device)
                self.memory = ReplayBuffer(agent.buffer_size, 
                                           (agent.input_frames, agent.input_dim, agent.input_dim), 
                                           agent.batch_size, 
                                           agent.input_feature_type)

        elif agent.is_noisy == False:
            print("NoisyNet is not applied.")
            self.epsilon_store = [] # a variable for notebook inline plotting

            if agent.input_feature_type == '1dim':
                print("The input feature is 1-dimension.")
                self.select_action = self.select_action_1dim
                self.q_current = QLinearNetwork(agent.input_dim, 
                                                agent.action_dim, 
                                                ).to(agent.device)
                self.q_target = QLinearNetwork(agent.input_dim, 
                                                agent.action_dim, 
                                                ).to(agent.device)
                self.memory = ReplayBuffer(agent.buffer_size, 
                                           agent.input_dim, 
                                           agent.batch_size, 
                                           agent.input_feature_type)

            elif agent.input_feature_type == '2dim':
                print("The input feature is 2-dimension.")
                self.select_action = self.select_action_2dim
                self.q_current = QConvNetwork((agent.input_frames, agent.input_dim, agent.input_dim), 
                                               agent.action_dim
                                               ).to(agent.device)
                self.q_target = QConvNetwork((agent.input_frames, agent.input_dim, agent.input_dim), 
                                               agent.action_dim
                                               ).to(agent.device)
                self.memory = ReplayBuffer(agent.buffer_size, 
                                           (agent.input_frames, agent.input_dim, agent.input_dim), 
                                           agent.batch_size, 
                                           agent.input_feature_type)

        if agent.input_feature_type == '1dim':
            self.get_init_state = self.get_init_state_1dim
            self.get_state = self.get_state_1dim
            self._compute_loss = self._compute_loss_1dim

        elif agent.input_feature_type == '2dim':
            self.get_init_state = self.get_init_state_2dim
            self.get_state = self.get_state_2dim
            self._compute_loss = self._compute_loss_2dim

    def select_action_1dim(self, agent, state):
        '''Must be pre-processed in the same way while updating current Q network. See def _compute_loss'''

        if np.random.random() < agent.epsilon:
            agent.epsilon = self._epsilon_step(agent.epsilon, agent.eps_decay)
            return np.zeros(agent.action_dim), agent.env.action_space.sample()
        else:
            state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
            Qs = self.q_current(state)
            action = Qs.argmax()
            agent.epsilon = self._epsilon_step(agent.epsilon, agent.eps_decay)
            return Qs.detach().cpu().numpy(), action.detach().item()

    def select_action_1dim_noisynet(self, agent, state):
        '''Must be pre-processed in the same way while updating current Q network. See def _compute_loss'''

        state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
        Qs = self.q_current(state)
        action = Qs.argmax()
        return Qs.detach().cpu().numpy(), action.detach().item()

    def select_action_2dim(self, agent, state):
        '''Must be pre-processed in the same way while updating current Q network. See def _compute_loss'''
        
        if np.random.random() < agent.epsilon:
            agent.epsilon = self._epsilon_step(agent.epsilon, agent.eps_decay)
            return np.zeros(agent.action_dim), agent.env.action_space.sample()
        else:
            state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)/255
            Qs = self.q_current(state)
            action = Qs.argmax()
            agent.epsilon = self._epsilon_step(agent.epsilon, agent.eps_decay)
            return Qs.detach().cpu().numpy(), action.detach().item()

    def select_action_2dim_noisynet(self, agent, state):
        '''Must be pre-processed in the same way while updating current Q network. See def _compute_loss'''
        
        state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)/255
        Qs = self.q_current(state)
        action = Qs.argmax()
        return Qs.detach().cpu().numpy(), action.detach().item()

    def get_init_state_1dim(self, agent):
        '''
        This method is for 1-dimensional state.
        input_dim: length of a state vector
        '''

        init_state = agent.env.reset()
        for _ in range(0): # Random initial starting point 
            action = agent.env.action_space.sample()
            init_state, _, _, _ = agent.env.step(action) 
        return init_state

    def get_init_state_2dim(self, agent):
        '''
        This method is for 2-dimensional state.

        input_frames: how many frames to be merged to a single state
        input_dim: hight and width of input resized image
        skipped_frame: how many frames to be skipped
        '''

        init_state = np.zeros((agent.input_frames, agent.input_dim, agent.input_dim))
        init_frame = agent.env.reset()
        init_state[0] = self.processing_resize_and_gray(init_frame, agent.input_dim)
        
        for i in range(1, agent.input_frames): 
            action =agent.env.action_space.sample()
            for j in range(agent.skipped_frame):
                state, _, _, _ = agent.env.step(action) 
            state, _, _, _ = agent.env.step(action) 
            init_state[i] = self.processing_resize_and_gray(state, agent.input_dim) 
        return init_state

    def get_state_1dim(self, agent, state, action):
        '''
        This method is for 1-dimensional state.
        input_dim: length of a state vector
        '''
        next_state, reward, done, _ = agent.env.step(action)
        return reward, next_state, done

    def get_state_2dim(self, agent, state, action):
        '''
        This method is for 2-dimensional state.

        input_frames: how many frames to be merged to a single state
        input_dim: hight and width of input resized image
        skipped_frame: how many frames to be skipped
        '''
        next_state = np.zeros((agent.input_frames, agent.input_dim, agent.input_dim))
        for i in range(len(state)-1):
            next_state[i] = state[i+1]

        rewards = 0
        dones = 0
        for _ in range(agent.skipped_frame):
            state, reward, done, _ = agent.env.step(action) 
            rewards += reward 
            dones += int(done) 
        state, reward, done, _ = agent.env.step(action) 
        next_state[-1] = self.processing_resize_and_gray(state, agent.input_dim) 
        rewards += reward 
        dones += int(done) 
        return rewards, next_state, dones

    def processing_resize_and_gray(self, state_image, input_dim):
        state_image = cv2.cvtColor(state_image, cv2.COLOR_RGB2GRAY) # No crop. Just getting a pure image from environment
        state_image = cv2.resize(state_image, dsize=(input_dim, input_dim)).reshape(input_dim, input_dim).astype(np.uint8)
        return state_image 

    def eval_and_save(self, agent, frame_idx, history_store, scores, avg_scores, execute_time):
        if np.mean(scores[-10:]) > max(avg_scores):
            torch.save(self.q_current.state_dict(), agent.model_path+'{}_Avg_Score_{}.pt'.format(frame_idx, np.mean(scores[-10:])))
            training_time = round((time.time()-execute_time)/3600, 1)
            np.save(agent.model_path+'{}_history_Score_{}_{}hrs.npy'.format(frame_idx, scores[-1], training_time), np.array(history_store, dtype=object))
            print("                    | Model saved. Recent scores: {}, Training time: {}hrs".format(scores[-10:], training_time), ' /'.join(os.getcwd().split('/')[-3:]))

        if scores[-1] > max(scores[:-1]):
            torch.save(self.q_current.state_dict(), agent.model_path+'{}_Highest_Score_{}.pt'.format(frame_idx, scores[-1]))
            training_time = round((time.time()-execute_time)/3600, 1)
            np.save(agent.model_path+'{}_Highest_Score_{}_{}hrs.npy'.format(frame_idx, scores[-1], training_time), np.array(history_store, dtype=object))
            print("                    | Model saved. Highest score: {}, Training time: {}hrs".format(scores[-1], training_time), ' /'.join(os.getcwd().split('/')[-3:]))
                
    def _epsilon_step(self, epsilon, eps_decay):
        ''' Epsilon decay control '''
        eps_decay_list = [eps_decay, eps_decay/2.5, eps_decay/3.5, eps_decay/5.5] 

        if epsilon>0.30:
            return max(epsilon-eps_decay_list[0], 0.1)
        elif epsilon>0.27:
            return max(epsilon-eps_decay_list[1], 0.1)
        elif epsilon>1.7:
            return max(epsilon-eps_decay_list[2], 0.1)
        else:
            return max(epsilon-eps_decay_list[3], 0.1)
    
    def plot(self, agent, frame_idx, score, scores, total_losses, epi_Qs, epi_loss):
        ''' Plotting current scores and relavant variables '''

        if agent.plot_option=='notebook': 
            total_losses.append(np.mean(epi_loss[-10:]))
            self.epsilon_store.append(agent.epsilon)
            self._plot(frame_idx, scores, total_losses, self.epsilon_store)
        
        elif agent.plot_option=='wandb': 
            wandb.log({'Score': score, 
                       'Episodic accumulated frames': len(epi_Qs), 
                       'Episode Count': len(scores)-1, 
                       'Episodic accumulated loss': epi_loss,
                       'Q(max_avg)': np.array(epi_Qs).max(1).mean(),
                       'Q(min_avg)': np.array(epi_Qs).min(1).mean(),
                       'Q(mean)': np.array(epi_Qs).flatten().mean(),
                       'Q(max)': np.array(epi_Qs).flatten().max(),
                       'Q(min)': np.array(epi_Qs).flatten().min()}, step=frame_idx)
            print(score, end='\r')

        elif (agent.plot_option=='wandb') & (agent.is_noisy == 'True'):
            wandb.log({'Score': scores[-1], 
                       'loss(10 frames avg)': np.mean(epi_loss[-10:])
                       })
            print(scores[-1], end='\r')

        else: 
            print(f'Episode Score: {scores[-1]}, Current Epsilon: {agent.epsilon}', end='\r')

    def _compute_loss_1dim(self, agent, batch: "Dictionary (S, A, R', S', Dones)"):
        states = torch.FloatTensor(batch['states']).to(agent.device)
        next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
        actions = torch.LongTensor(batch['actions'].reshape(-1, 1)).to(agent.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(agent.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(agent.device)

        current_q = self.q_current(states).gather(1, actions)
        # The next line is the same calculation in the Double DQN.
        next_q = self.q_target(next_states).gather(1, self.q_current(next_states).argmax(axis=1, keepdim=True)).detach()
        mask = 1 - dones
        target = (rewards + (mask * agent.gamma * next_q)).to(agent.device)

        loss = F.smooth_l1_loss(current_q, target)
        return loss

    def _compute_loss_2dim(self, agent, batch: "Dictionary (S, A, R', S', Dones)"):
        # If normalization is used, it must be applied to 'state' and 'next_state' here. ex) state/255
        states = torch.FloatTensor(batch['states']).to(agent.device) / 255
        next_states = torch.FloatTensor(batch['next_states']).to(agent.device) / 255
        actions = torch.LongTensor(batch['actions'].reshape(-1, 1)).to(agent.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(agent.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(agent.device)

        current_q = self.q_current(states).gather(1, actions)
        # The next line is the same calculation in the Double DQN.
        next_q = self.q_target(next_states).gather(1, self.q_current(next_states).argmax(axis=1, keepdim=True)).detach()
        mask = 1 - dones
        target = (rewards + (mask * agent.gamma * next_q)).to(agent.device)

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