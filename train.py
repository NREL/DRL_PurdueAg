import os
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
from ray import tune
import rom
import utils
import ray
import shutil
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from controller_env import OptimControllerEnv
import time

def main():
    start_time = time.time()

    env = OptimControllerEnv()
    obs = env.reset()
    print("Checking the environment ... \n")
    while True:
        action = env.action_space.sample()
        exogenous_variable = env.commanded_RPM
        obs, r, done, _ = env.step(action=action)
        if r > -2000:
            print(r, action)
            print(obs.flatten()) 
        if done:
            break

    # Register the environment

    def env_creator(env_config={}):
        return OptimControllerEnv()  # returns an env instance

    register_env("my_env", env_creator)

    # Run the agent

    # res = tune.run("PPO",
    #         stop={
    #         'timesteps_total': 100000},
    #          config = {"env": "my_env",
    #                    "num_workers": 1,
    #                    "vf_clip_param":1E6},
    #          local_dir="./Agent_experiment/",
    #          log_to_file=True)

    chkpt_root = "tmp/PPO"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config['vf_clip_param'] = 1e5
    config['evaluation_num_episodes'] = 10000
    agent = ppo.PPOTrainer(config, env="my_env")
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 250
    training_result = {}
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        training_result[n+1] = result
        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))

    comp_time_seconds = time.time()-start_time
    print("Total time (seconds) %f"%comp_time_seconds)
    print("Total computation time: %d hours %d \
    minutes %f seconds"%(comp_time_seconds // 3600, (comp_time_seconds % 3600) \
    //60, comp_time_seconds%60))
    return training_result
    

if __name__ == "__main__":
    training_result = main()
    pd.DataFrame(training_result).to_csv(str(time.time())+'-trainingresult')
    



