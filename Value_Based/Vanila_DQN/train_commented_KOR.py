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
from replay_buffer import ReplayBuffer
from qnetwork import QNetwork 

import matplotlib.pyplot as plt
from IPython.display import clear_output

import wandb   

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
env_name = env_list[6]
env = gym.make(env_name)
input_dim = 84
input_frame = 4
print("env_name", env_name) 
print(env.unwrapped.get_action_meanings(), env.action_space.n) 

update_start_buffer_size = 10000 # Buffer에 random action의 결과를 계속해서 쌓는 것을 언제까지 할지
num_frames = 10000000 # 총 학습 frame 수 (= 학습할 동안 agent가 action을 행하는 횟수) 

# e-Greedy exploration 변수. 
eps_max = 1.0  
eps_min = 0.1  
eps_decay = 1/1200000 
# 위의 3개는 얼마나 agent가 greedy하게 움직이면서 exploration을 할지 정해주는 매개변수

gamma = 0.99 # Discounted factor. 더 먼 미래에 받을 보상일 수록 얼마나 discount할지를 정해주는 변수 

buffer_size = int(8e5) # Replay Buffer 사이즈
batch_size = 32        # Replay Buffer에서 한번에 뽑아서 가져오는 sample의 개수    
update_type = 'hard'   # Q-Network을 업데이트할 때, 어떤 방식으로 할지 정하는 것. 일반 DQN 논문들은 일반적으로 hard를 함.
target_update_freq = 250 # Update type이 hard일때, 몇 frame마다 update를 할지 정하는 변수. 
soft_update_tau = 0.002  # Update type이 soft일때, target network의 변수를 가져오는 비율.
learning_rate = 0.0001 
skipped_frame = 4 # 환경에서 주는 frame을 skip하는 것이 필요한 경우. 

# Customized Parameters. GPU device 및 모델 저장 경로등을 설정하는 부분 
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

# Trained model 의 경로 지정
trained_model_path = ''

# Plot 옵션 선택. 1) wandb 이용 2) inline: 쥬피터 안에서 plot 3) False: plot안하기 
plot_options = {0: 'wandb', 1: 'inline', 2: False} 
plot_option = plot_options[0]

if plot_option=='wandb':
    call(["wandb", "login", "000c1d3d8ebb4219c3a579d5ae02bc38be380c70"])
    os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
    wandb_project_name = 'temporary'
    wandb.init(
            project=wandb_project_name,
            name=f"{rand_name}_{folder_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_frame": input_frame,
                    "input_dim": input_dim,
                    "eps_max": eps_max,
                    "eps_min": eps_min,
                    "eps_decay": eps_decay,
                    "num_frames": num_frames,
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
    num_frames,
    skipped_frame,
    eps_decay,
    gamma,
    target_update_freq,
    update_type,
    soft_update_tau,
    batch_size,
    buffer_size,
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