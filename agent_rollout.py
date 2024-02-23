import os
import gym
from gym import spaces
import rom
import utils
import ray
import shutil
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from controller_env import OptimControllerEnv
from gym.envs.registration import register
import time
import numpy as np
import matplotlib.pyplot as plt
from plot import Plot

register(
    id='OptimController-v0',
    entry_point = 'controller_env:OptimControllerEnv'
)

def env_creator(env_config={}):
    return OptimControllerEnv()  # returns an env instance

register_env("OptimController-v0", env_creator)

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
config['vf_clip_param'] = 1e5
config['evaluation_num_episodes'] = 10000
agent = ppo.PPOTrainer(config, env="OptimController-v0")

# examine the trained policy
policy = agent.get_policy()
model = policy.model


# # apply the trained policy in a rollout
chkpt_file = "tmp/PPO/checkpoint_000250/checkpoint-250"
agent.restore(chkpt_file)
env = gym.make("OptimController-v0")

state = env.reset()
sum_reward = 0
n_step = 300

# print(state)

# compute action of the trained agent given the state

action = agent.compute_action(state)
# print(action)
state_list = []
action_list = []
reward_list = []
cmd_rpm = []
for step in range(n_step):
    action = agent.compute_action(state)
    state, reward, done, info = env.step(action)
    state_list.append(state)
    action_list.append(info['Action'])
    reward_list.append(reward)
    cmd_rpm.append(info['Commanded RPM'])

    # env.render()

    if done == 1:
        # report at the end of each episode
        # print("cumulative reward", sum_reward)
        state = env.reset()
        # sum_reward = 0
state_label =  ['Bulk_P', 'Alt_P', 'Vac_P', 'Fert_P', 
                'Bulk_Q', 'Alt_Q', 'Vac_Q', 'Fert_Q',
                'Bulk_rpm_delta', 'Alt_rpm_delta', 'Vac_rpm_delta', 
                'Fert_rpm_delta']

cmdRPM_label = ['Bulk_rpm_cmd', 'Alt_rpm_cmd', 
                'Vac_rpm_cmd', 'Fert_rpm_cmd']
action_label = ['pHP', 'pMP', 'Bulk_mode', 
                'Alt_mode', 'Vac_mode', 'Fert_mode']

state_array = np.array(state_list).reshape(n_step, len(state_label))
action_array = np.array(action_list)
reward_array = np.array(reward_list)
cmdRPM_array = np.array(cmd_rpm).reshape(n_step, len(cmdRPM_label))



state_dict = {}
action_dict = {}
cmdRPM_dict = {}

# Saving state vectors in a dictionary
for index, label in enumerate(state_label):
    state_dict[label] = state_array[:,index]

# Saving action vectors in a dictionary
pHP = []
pMP = []
Bulk_mode = []
Alt_mode = []
Vac_mode = []
Fert_mode = []
for i in range(n_step):
    pHP.append(action_array[i]['continuous'].flatten()[0])
    pMP.append(action_array[i]['continuous'].flatten()[1])
    Bulk_mode.append(action_array[i]['discrete'].flatten()[0])
    Alt_mode.append(action_array[i]['discrete'].flatten()[1])
    Vac_mode.append(action_array[i]['discrete'].flatten()[2])
    Fert_mode.append(action_array[i]['discrete'].flatten()[3])
       
action_list = [pHP,pMP,Bulk_mode,Alt_mode,Vac_mode,Fert_mode]
for index, label in enumerate(action_label):
    action_dict[label] = np.array(action_list[index])

# Saving cmd rpm in a dictionary
for index, label in enumerate(cmdRPM_label):
    cmdRPM_dict[label] = cmdRPM_array[:,index]

# plot results
    
Plot(action_dict).plot_action()
# Plot(action_dict).plot_action(variables=['Bulk_mode', 
#                 'Alt_mode', 'Vac_mode', 'Fert_mode'])
Plot(action_dict).plot_action(plot_modes=True)
Plot(state_dict).plot_state(pressures=True)
Plot(state_dict).plot_state(flowrates=True)
Plot(state_dict).plot_state(RPM=True)
Plot(cmdRPM_dict).plot_cmdRPM()

