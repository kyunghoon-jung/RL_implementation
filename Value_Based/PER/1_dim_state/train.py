import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torchsummary import summary
import numpy as np
import time    
import gym    
import cv2
import os

from agent import Agent
from replay_buffer import PrioritizedReplayBuffer
from qnetwork import QNetwork 

import matplotlib.pyplot as plt
from IPython.display import clear_output

import wandb   

# env = gym.make('CartPole-v0')
# env = gym.make('LunarLander-v2')
# env = gym.make('Breakout-v0')
# env = gym.make('BreakoutDeterministic-v4')
# env = gym.make('BoxingDeterministic-v4')
# env = gym.make('BreakoutNoFrameskip-v4')
# env = gym.make('PongDeterministic-v4')  
# env_name = 'Boxing-'+version
# env.seed(0)
env_name = 'CartPole'
env_version = 0
env_name = env_name+'-v'+str(env_version)
env = gym.make(env_name)
input_dim = 84
input_frame = 4
input_type = '1-dim'
if input_type=='1-dim':
    input_dim = env.observation_space.shape[0]
print("env_name", env_name) 
# print(env.unwrapped.get_action_meanings(), env.action_space.n) 

update_start_buffer_size = 100
num_frames = 10000
eps_max = 1.0
eps_min   = 0.1
eps_decay = 1/2000
gamma = 0.99

buffer_size = int(3000) 
batch_size = 32
update_type = 'hard'
soft_update_tau = 0.002
learning_rate = 0.001
target_update_freq = 100
skipped_frame = 0

#for PER
alpha = 0.5
beta = 0.6
epsilon_for_priority = 1e-6

device_num = 0
rand_seed = None
rand_name = ('').join(map(str, np.random.randint(10, size=(3,))))
folder_name = os.getcwd().split('/')[-1] 

model_number = 0
model_save_path = \
f'./model_save/{rand_name}_{env_name}_num_f:{num_frames}_eps_dec:{round(eps_decay,8)}f\
_gamma:{gamma}_tar_up_frq:{target_update_freq}f\
_up_type:{update_type}_soft_tau:{soft_update_tau}f\
_batch:{batch_size}_buffer:{buffer_size}f\
_up_start:{update_start_buffer_size}_lr:{learning_rate}f\
_eps_min:{eps_min}_device:{device_num}_rand:{rand_seed}_{model_number}/'
if not os.path.exists('./model_save/'):
    os.mkdir('./model_save/')
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
print("model_save_path:", model_save_path)

# plot_option = 'wandb'
# plot_option = 'inline'
plot_option = False

if plot_option=='wandb':
    os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
    wandb_project_name = 'per_test'
    wandb.init(
            project=wandb_project_name,
            name=f"{rand_name}_{folder_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_frame": input_frame,
                    "input_dim": input_dim,
                    "input_type": input_type,
                    "eps_max": eps_max,
                    "eps_min": eps_min,
                    "eps_decay": eps_decay,
                    "num_frames": num_frames,
                    "alpha (PER)": alpha,
                    "beta (PER)": beta,
                    "epsilon_for_priority (PER)": epsilon_for_priority,
                    "skipped_frame": skipped_frame,
                    "gamma": gamma,
                    "buffer_size": buffer_size,
                    "update_start_buffer_size": update_start_buffer_size,
                    "batch_size": batch_size,
                    "update_type": update_type,
                    "soft_update_tau": soft_update_tau,
                    "learning_rate": learning_rate,
                    "target_update_freq (unit:frames)": target_update_freq,
                    }
            )

agent = Agent( 
    env,
    input_frame,
    input_dim,
    input_type,
    num_frames,
    skipped_frame,
    eps_decay,
    gamma,
    target_update_freq,
    update_type,
    soft_update_tau,
    batch_size,
    buffer_size,
    alpha,
    beta,
    epsilon_for_priority,
    update_start_buffer_size,
    learning_rate,
    eps_min,
    eps_max,
    device_num,
    rand_seed,
    plot_option,
    model_save_path
) 

agent.train()