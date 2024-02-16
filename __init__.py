from gym.envs.registration import register

register(
    id='OptimController-v0',
    entry_point = 'controller_env:OptimControllerEnv'
)