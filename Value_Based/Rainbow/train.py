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

from agent import Agent
from replay_buffer import ReplayBuffer
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
    "SeaquestDeter": "SeaquestDeterministic-v4",
    "QbertDeter": "QbertDeterministic-v4",
    "SpaceInvDeter": "SpaceInvadersDeterministic-v4",
    "PhoenixDeter": "PhoenixDeterministic-v4",
    "FreewayDeter": "FreewayDeterministic-v4",
    "AssaultDeter": "AssaultDeterministic-v4",
    "MontezumaDeter":"MontezumaRevengeDeterministic-v4"
}
env_name = env_list["SeaquestDeter"]
env = gym.make(env_name)
input_dim = 84
input_frame = 4 
print("env_name", env_name) 
print(env.unwrapped.get_action_meanings(), env.action_space.n)

update_start_buffer_size = 20000 # 80K frames in the Rainbow paper
tot_train_frames = 50000000

gamma = 0.99
buffer_size = int(7e5) 
batch_size = 32 
update_type = 'hard'
soft_update_tau = 0.002
learning_rate = 0.00025 / 4
target_update_freq = 8000 # Update frequency of target Q-Network. Same as in Rainbow paper(24000 frames=8000 steps).
behave_update_freq = 4 # Update frequency of current Q-Network.  
grad_clip = 10
skipped_frame = 0

device_num = 1
rand_seed = None
rand_name = ('').join(map(str, np.random.randint(10, size=(3,))))
folder_name = os.getcwd().split('/')[-1] 

# NoisyNet Variable
initial_std = 0.5 

# Variables for Multi-step TD
n_step = 3 # the best step size in the Rainbow paper

# Variables for Categorical RL
n_atoms = 51
Vmax = 10
Vmin = -10

#In PER paper, alpha=0.6, beta=0.4 for the propotional variant (alpha=0.7, beta=0.5 for the rank-based variant). These are choosen hueristically.
alpha = 0.6
beta = 0.4
epsilon_for_priority = 1e-6
trained_model_path = ''

project_name = 'rainbow'
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

plot_option = 'wandb'
# plot_option = 'inline'
is_render = True
if is_render:
    vis_env_id = f'{rand_name}_{env_name}'
else: vis_env_id = ''

if plot_option=='wandb':
    call(["wandb", "login", "000c1d3d8ebb4219c3a579d5ae02bc38be380c70"])
    os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
    wandb.init(
            project=project_name,
            name=f"{rand_name}_{folder_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_frame": input_frame,
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
                    "grad_clip": grad_clip,
                    "batch_size": batch_size,
                    "update_type": update_type,
                    "soft_update_tau": soft_update_tau,
                    "learning_rate": learning_rate,
                    "target_update_freq (unit:frames)": target_update_freq,
                    "behave_update_freq (unit:frames)": behave_update_freq,
                    "n_atoms (C51 param)": n_atoms,
                    "Vmax (C51 param)": Vmax,
                    "Vmin (C51 param)": Vmin
                    }
            )

agent = Agent( 
    env,
    input_frame,
    input_dim,
    initial_std,
    tot_train_frames,
    skipped_frame,
    gamma,
    alpha,
    beta,
    epsilon_for_priority,
    n_step,
    Vmax,
    Vmin,
    n_atoms,
    target_update_freq,
    behave_update_freq,  
    grad_clip,
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
    trained_model_path,
    is_render,
    vis_env_id
) 

agent.train()