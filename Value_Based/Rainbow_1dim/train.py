import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torchsummary import summary
from subprocess import call
import numpy as np
import time    
import gym    
import cv2
import os
from adabelief_pytorch import AdaBelief

from agent import Agent
from replay_buffer import ReplayBuffer
from qnetwork import QNetwork 
import matplotlib.pyplot as plt
from IPython.display import clear_output
import wandb   
from subprocess import call
call(["wandb", "login", "000c1d3d8ebb4219c3a579d5ae02bc38be380c70"])

env_list = {
    0: "CartPole-v0",
    1: "CartPole-v2",
    2: "LunarLander-v2",
    3: "Breakout-v4",
    4: "BreakoutDeterministic-v4",
    5: "BreakoutNoFrameskip-v4",
    6: "BoxingDeterministic-v4",
    7: "PongDeterministic-v4",
}
# env.seed(0)
env_name = env_list[0]
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
print("env_name", env_name) 
print(env.action_space.n) 

update_start_buffer_size = 100
tot_train_frames = 20000

gamma = 0.99
buffer_size = int(1000) 
batch_size = 32           
update_type = 'hard'
soft_update_tau = 0.002
learning_rate = 0.001
target_update_freq = 100
current_update_freq = 1 # Update frequency of current Q-Network.  
skipped_frame = 0

device_num = 1
rand_seed = None
rand_name = ('').join(map(str, np.random.randint(10, size=(3,))))
folder_name = os.getcwd().split('/')[-1] 

# NoisyNet Variable
initial_std = 0.5 

# Variables for Multi-step TD
n_step = 3

# Variables for Categorical RL
n_atoms = 51
Vmax = 10
Vmin = -10

#In PER paper, alpha=0.6, beta=0.4 for the propotional variant (alpha=0.7, beta=0.5 for the rank-based variant). These are choosen hueristically.
alpha = 0.2
beta = 0.6
epsilon_for_priority = 1e-6

model_number = 0
model_save_path = \
f'./model_save/{rand_name}_{env_name}_tot_f:{tot_train_frames}f\
_gamma:{gamma}_tar_up_frq:{target_update_freq}f\
_up_type:{update_type}_soft_tau:{soft_update_tau}f\
_batch:{batch_size}_buffer:{buffer_size}f\
_up_start:{update_start_buffer_size}_lr:{learning_rate}f\
_device:{device_num}_rand:{rand_seed}_{model_number}/'
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
    wandb_project_name = 'rainbow-per'
    wandb.init(
            project=wandb_project_name,
            name=f"{rand_name}_{folder_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_dim": input_dim,
                    "alpha": alpha,
                    "beta": beta,
                    "epsilon_for_priority": epsilon_for_priority,
                    "initial_std (NoisyNet param)": initial_std,
                    "total_training_frames": tot_train_frames,
                    "skipped_frame": skipped_frame,
                    "gamma": gamma,
                    "n_step (Multi-step param)": n_step,
                    "buffer_size": buffer_size,
                    "update_start_buffer_size": update_start_buffer_size,
                    "batch_size": batch_size,
                    "update_type": update_type,
                    "soft_update_tau": soft_update_tau,
                    "learning_rate": learning_rate,
                    "target_update_freq (unit:frames)": target_update_freq,
                    "n_atoms (C51 param)": n_atoms,
                    "Vmax (C51 param)": Vmax,
                    "Vmin (C51 param)": Vmin
                    }
            )

agent = Agent( 
    env,
    input_dim,
    initial_std,
    tot_train_frames,
    skipped_frame,
    gamma,
    alpha,
    beta,
    epsilon_for_priority,
    n_step,
    target_update_freq,
    current_update_freq,  
    update_type,
    soft_update_tau,
    batch_size,
    buffer_size,
    update_start_buffer_size,
    learning_rate,
    device_num,
    rand_seed,
    plot_option,
    model_save_path,
    n_atoms,
    Vmax,
    Vmin
) 

agent.train()