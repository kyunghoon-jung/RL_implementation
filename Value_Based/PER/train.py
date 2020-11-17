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
from subprocess import call

env_list = {
    "Cart_1": "CartPole-v0",
    "Cart_2": "CartPole-v2",
    "Lunar": "LunarLander-v2",
    "Breakout": "Breakout-v4",
    "BreakoutDeter": "BreakoutDeterministic-v4",
    "BreakoutNo": "BreakoutNoFrameskip-v4",
    "BoxingDeter": "BoxingDeterministic-v4",
    "PongDeter": "PongDeterministic-v4",
    "Seaquest": "SeaquestDeterministic-v4",
    "QbertDeter": "QbertDeterministic-v4",
    "SpaceInvDeter": "SpaceInvadersDeterministic-v4",
    "PhoenixDeter": "PhoenixDeterministic-v4",
    "FreewayDeter": "FreewayDeterministic-v4",
    "AssaultDeter": "AssaultDeterministic-v4",
    "MontezumaDeter":"MontezumaRevengeDeterministic-v4",
    "GopherDeter":"GopherDeterministic-v4"
}
env_name = env_list["GopherDeter"]
env = gym.make(env_name)
input_dim = 84
input_frame = 4
print("env_name", env_name) 
print(env.unwrapped.get_action_meanings(), env.action_space.n) 

update_start_buffer_size = 20000
tot_train_frames = 50000000
eps_max = 1.0
eps_min = 0.1
eps_decay = 1/1250000
gamma = 0.99

buffer_size = int(8e5) 
batch_size = 32
update_type = 'hard'
soft_update_tau = 0.001
learning_rate = 0.00025 / 4 # In PER paper, they applied 1/4 of the learning_rate(=step size) in vanila DQN paper. Due to relatively higher gradient magnitudes. 
current_update_freq = 4 # Update frequency of current Q-Network.  
target_update_freq = 8000
skipped_frame = 0

#In PER paper, alpha=0.6, beta=0.4 for the propotional variant (alpha=0.7, beta=0.5 for the rank-based variant). These are choosen hueristically.
alpha = 0.6
beta = 0.4
epsilon_for_priority = 1e-6

device_num = 1
rand_seed = None
rand_name = ('').join(map(str, np.random.randint(10, size=(3,))))
folder_name = os.getcwd().split('/')[-1] 

project_name = 'per-test'
model_number = 0
main_path = '/data3/Jungkh/RL/'
model_save_path = \
f'{rand_name}_{folder_name}_{env_name}_tot_f:{tot_train_frames}f\
_gamma:{gamma}_tar_up_frq:{target_update_freq}f\
_up_type:{update_type}_soft_tau:{soft_update_tau}f\
_batch:{batch_size}_buffer:{buffer_size}f\
_up_start:{update_start_buffer_size}_lr:{learning_rate}f\
_device:{device_num}_rand:{rand_seed}_{model_number}/'
if not os.path.exists(main_path):
    if not os.path.exists('./model_save/'):
        os.mkdir('./model_save/')
    model_save_path = './model_save/' + model_save_path
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
else:
    model_save_path = main_path + model_save_path
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
print("model_save_path:", model_save_path)
trained_model_path = ''
if trained_model_path:
    print("Trained Model Path: ", trained_model_path)

plot_option = 'wandb'
# plot_option = 'inline'
# plot_option = False

if plot_option=='wandb':
    call(["wandb", "login", "000c1d3d8ebb4219c3a579d5ae02bc38be380c70"])
    os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
    wandb.init(
            project=project_name,
            name=f"{rand_name}_{folder_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_frame": input_frame,
                    "input_dim": input_dim,
                    "eps_max": eps_max,
                    "eps_min": eps_min,
                    "eps_decay": eps_decay,
                    "tot_train_frames": tot_train_frames,
                    "alpha (PER)": alpha,
                    "beta (PER)": beta,
                    "epsilon_for_priority (PER)": epsilon_for_priority,
                    "skipped_frame": skipped_frame,
                    "gamma": gamma,
                    "buffer_size": buffer_size,
                    "current_update_freq": current_update_freq,
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
    tot_train_frames,
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
    current_update_freq,
    update_start_buffer_size,
    learning_rate,
    eps_min,
    eps_max,
    device_num,
    rand_seed,
    plot_option,
    model_save_path,
    trained_model_path
) 

agent.train()
