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
chkpt_file = "tmp/PPO/checkpoint_000050/checkpoint-50"
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
for step in range(n_step):
    action = agent.compute_action(state)
    state, reward, done, info = env.step(action)
    state_list.append(state)
    action_list.append(action)
    sum_reward += reward

    # env.render()

    if done == 1:
        # report at the end of each episode
        print("cumulative reward", sum_reward)
        state = env.reset()
        sum_reward = 0