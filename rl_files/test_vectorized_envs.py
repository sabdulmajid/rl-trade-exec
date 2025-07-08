import sys
import os
# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the sys path
sys.path.append(parent_dir)
from simulation.market_gym import Market
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np


if __name__ == '__main__':
    config = {'market_env': 'flow', 'execution_agent': 'rl_agent', 'volume': 10, 'seed': 0}
    config1 = {'market_env': 'flow', 'execution_agent': 'rl_agent', 'volume': 10, 'seed': 1}
    config2 = {'market_env': 'flow', 'execution_agent': 'rl_agent', 'volume': 10, 'seed': 2}
    # config = {'market_env': 'flow', 'execution_agent': 'rl_agent', 'volume': 10, 'seed': 0}
    # M = Market(config)

    env_fns = [lambda: Market(config), lambda: Market(config1), lambda: Market(config1), lambda: Market(config2)]

    env = AsyncVectorEnv(env_fns=env_fns)
    # env = SyncVectorEnv(env_fns=env_fns)

    obs = env.reset()
    print(obs)

    for n in range(10):    
        # action = env.action_space.sample()
        a = np.array([-10,10,-10,-10,-10], dtype=np.float32)
        action = np.repeat(a[np.newaxis, :], 4, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        print(terminated)
        if terminated.any():
            pass

    print(obs)
    print(info)
    print(info['final_info'])
    print(info['final_observation'])


    """
    - obs.shape = (n_envs, n_observations)
    - reward.shape = (n_envs,)
    - terminated.shape = (n_envs,)
    - truncated.shape = (n_envs,) 
    - environments auto reset, once they reach terminal state 
    - when the environment auto resets, info has additional keys 'final_obersevation' and 'final_info'
    """

