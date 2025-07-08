import os, sys 
current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from simulation.agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent, SubmitAndLeaveAgent, MarketAgent, InitialAgent, ObservationAgent, RLAgent
from limit_order_book.limit_order_book import LimitOrderBook
from config.config import noise_agent_config, strategic_agent_config, sl_agent_config, linear_sl_agent_config, market_agent_config, initial_agent_config, observation_agent_config, rl_agent_config
import numpy as np
import pandas as pd 
from config.config import noise_agent_config
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from multiprocessing import Pool
import itertools
import time
import gymnasium as gym 
import torch

class Market(gym.Env):
    ''''
    This is a gym environment which replicates a market simulation

    Attributes:
        agents (dict): dictionary of agents in the market
        execution_agent_id (str): id of the execution agent (rl_agent, sl_agent, linear_sl_agent)
        lob (LimitOrderBook): the current state of the limit order book, gets modified every time orders are processed      
        pq (PriorityQueue): priority queue of events, gets updated every time an event is processed
    '''

    def __init__(self, config):

        """
        initialize the class with a config with keys 
            market_env (str): noise, flow, strategic
            execution_agent (str): sl_agent, linear_sl_agent, rl_agent
            volume (int): number of lots to trade
            seed (int): seed for the random number generator
            terminal_time (int): time at which the simulation should terminate
            time_delta (int): time difference at which the agent trades                                   
        """

        assert 'market_env' in config
        assert 'execution_agent' in config
        assert 'volume' in config
        assert 'seed' in config
        assert 'terminal_time' in config 
        assert 'time_delta' in config         

        seed = config['seed']
        
        assert config['market_env'] in ['noise', 'flow', 'strategic']
        assert config['execution_agent'] in ['market_agent', 'sl_agent', 'linear_sl_agent', 'rl_agent']

        self.agents = {}
        
        # set up initial agent          
        if config['market_env'] == 'noise':
            # those files need to be generated in advance. 0.65 denotes the damping factor 
            initial_agent_config['initial_shape_file'] = f'{parent_dir}/initial_shape/noise_65.npz'
            initial_agent_config['initial_shape_file'] = f'{parent_dir}/initial_shape/noise_65.npz'
        else:
            initial_agent_config['initial_shape_file'] = f'{parent_dir}/initial_shape/noise_flow_65.npz'
        initial_agent_config['start_time'] = -config['time_delta']
        agent = InitialAgent(**initial_agent_config)
        self.agents[agent.agent_id] = agent

        # set up the noise agent, general settings 
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        noise_agent_config['terminal_time'] = config['terminal_time']
        # always start the noise agent at -time_delta in all three cases 
        noise_agent_config['start_time'] = -config['time_delta']

        # set up noise agent for noise, flow, or strategic case 
        # noise  
        if config['market_env'] == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            agent = NoiseAgent(**noise_agent_config)
            self.agents[agent.agent_id] = agent

        # flow or strategic 
        else: 
            noise_agent_config['imbalance_reaction'] = True
            # can adjust imbalance_factor here 
            # noise_agent_config['imbalance_factor'] = 2.0
            agent = NoiseAgent(**noise_agent_config)            
            # TODO: make those intensity adjustments automatically or move them to the config file 
            # scale down other intensities due to presence of tactical traders 
            # we still keep this explicit, to remember that we do this kind of scaling 
            agent.limit_intensities = agent.limit_intensities * noise_agent_config['intensity_scaling']            
            agent.market_intensity = agent.market_intensity * noise_agent_config['intensity_scaling']
            agent.cancel_intensities = agent.cancel_intensities * noise_agent_config['intensity_scaling']
            self.agents[agent.agent_id] = agent

        # strategic agent 
        if config['market_env'] == 'strategic':       
            strategic_agent_config['terminal_time'] = config['terminal_time']
            strategic_agent_config['start_time'] = -config['time_delta']     
            # we use the default config settings. we can modify them as follows:  
            # strategic_agent_config['time_delta'] = 3 
            # strategic_agent_config['market_volume'] = 1
            # strategic_agent_config['limit_volume'] = 2
            strategic_agent_config['rng'] = np.random.default_rng(seed)
            agent = StrategicAgent(**strategic_agent_config)
            self.agents[agent.agent_id] = agent 
            # adjust start time for noise agent 
            # this might be redundant (probably already in the config) 
            # in the new setting we always start at -time_delta even in the noise and flow settings 
            # self.agents['noise_agent'].start_time = -config['time_delta']
            # self.agents['initial_agent'].start_time = -config['time_delta']

        # execution agent
        if config['execution_agent'] == 'market_agent':
            # sl_agent_config['start_time'] = 0
            market_agent_config['volume'] = config['volume']
            agent = MarketAgent(**market_agent_config)
        elif config['execution_agent'] == 'sl_agent':
            # sl_agent_config['start_time'] = 0
            sl_agent_config['volume'] = config['volume']
            sl_agent_config['terminal_time'] = config['terminal_time']
            agent = SubmitAndLeaveAgent(**sl_agent_config)
        elif config['execution_agent'] == 'linear_sl_agent': 
            # linear_sl_agent_config['start_time'] = 0
            linear_sl_agent_config['volume'] = config['volume']
            linear_sl_agent_config['terminal_time'] = config['terminal_time']
            linear_sl_agent_config['time_delta'] = config['time_delta']
            agent = LinearSubmitLeaveAgent(**linear_sl_agent_config)
        else:
            # rl_agent_config['start_time'] = 0
            rl_agent_config['terminal_time'] = config['terminal_time']
            rl_agent_config['time_delta'] = config['time_delta']
            rl_agent_config['volume'] = config['volume']
            if config['market_env'] == 'noise':
                rl_agent_config['initial_shape_file'] = f'{parent_dir}/initial_shape/noise_65.npz'
            else:
                rl_agent_config['initial_shape_file'] = f'{parent_dir}/initial_shape/noise_flow_65.npz'
            agent = RLAgent(**rl_agent_config)

        self.agents[agent.agent_id] = agent
        self.execution_agent_id = agent.agent_id

        if config['execution_agent'] == 'rl_agent':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(agent.observation_space_length,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-10, high=10, shape=(agent.action_space_length,), dtype=np.float32)    
            # observation agent interrupts the queue whenever an observation happens 
            agent = ObservationAgent(**observation_agent_config)
            self.agents[agent.agent_id] = agent
        else:
            # this is just dummy observation space, to make vectorized environments work for benchmark agents 
            # at the moment we are not using vectorized environments for the benchmarks, we rollout one environment per CPU
            # the latter is faster than vectorized environents 
            # vectorized environments are only used in RL training 
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # this is a bit hacky. adding a property to transform actions to the simplex space (this is mainly used for the multivariate normal distributions)
        if 'transform_action' in config:
            assert config['transform_action'] == True, 'transform_action should be True if contained in the config'                        
            self.transform_action = True
        else:
            self.transform_action = False

        return None 


    def reset(self, seed=None, options=None):
        self.lob = LimitOrderBook(list_of_agents=list(self.agents.keys()), level=30, only_volumes=False)
        # reset agents 
        for agent_id in self.agents:
            self.agents[agent_id].reset()
        # initialize event queue 
        self.pq = PriorityQueue()
        # set initial events 
        for agent_id in self.agents:
            # noise agent just puts the first event at its initial time (the first event should also be drawn from an exponential distribution)
            # although this is not completely correct 
            out = self.agents[agent_id].initial_event()
            self.pq.put(out)
        # runs up to first observation or stops if the simulation is terminated 
        # if there is no RL agent present and no observation agent present, transition will just run straight to the end without any intermediate action
        observation, reward, terminated, info = self.transition()
        # this will terminate if the execution agent is one of the benchmark agents 
        return observation, info
    
    def step(self, action=None):
        observation, reward ,terminated, info = self.transition(action)
        if terminated:
            assert self.agents[self.execution_agent_id].volume == 0
        return observation, reward, terminated, False, info 

    def transition(self, action=None):
        terminated = False
        transition_reward = 0 
        if action is not None:
            if self.transform_action:
                action = np.exp(action) / np.sum(np.exp(action))
        # n_events = 0  
        while not self.pq.empty(): 
            # get next event from the event queue 
            t, priority, agent_id = self.pq.get()
            if t > self.agents[self.execution_agent_id].terminal_time:
                # simulation should terminate at the execution agents terminal time
                raise ValueError("time is greater than execution agents terminal time")
            if agent_id == 'rl_agent':
                # if rl agent is present, generate an order based, the agent used information from the LOB 
                # orders could be None if agent doesnt change the current orders, but just leaves them in place 
                orders = self.agents[agent_id].generate_order(lob=self.lob, time=t, action=action)
            else:
                # this is either benchmark agent or observation agent 
                # the observation agent does not return any orders 
                # for the benchmark agents, no action is needed 
                orders = self.agents[agent_id].generate_order(lob=self.lob, time=t)
                # assert orders is not None
            # update order book, and check whether execution agent orders have been filled 
            # when can orders be None? 
            # rl agent doesnt change anything. just leaves orders where they are. any other reason should not occur !
            # note that rl agents returns empty list if it doesnt change anything
            if orders is not None or orders == []:
                msgs = self.lob.process_order_list(orders)
                # update execution agent position 
                # noise agent and strategic agent do not update their positions
                reward, terminated = self.agents[self.execution_agent_id].update_position_from_message_list(msgs)
                transition_reward += reward
                if terminated:
                    break
            # if not terminated or execution agent not present, generate a new event 
            # can be None if there are no more events happening for the agent 
            out = self.agents[agent_id].new_event(t, agent_id)
            if out is not None:
                self.pq.put(out)
            # observation agent breaks the loop 
            if agent_id == 'observation_agent':
                break
        # consitensy check: if terminated the execution agent's volume must be zero 
        if terminated:
            assert self.agents[self.execution_agent_id].volume == 0
        # TODO: could only record final info to increase speed 
        # record a bunch of infos 
        mid_price = (self.lob.data.best_bid_prices[-1] + self.lob.data.best_ask_prices[-1])/2
        initial_mid_price = (self.agents['initial_agent'].initial_ask + self.agents['initial_agent'].initial_bid)/2
        info = {'reward': self.agents[self.execution_agent_id].cummulative_reward, 
                'passive_fill_rate': self.agents[self.execution_agent_id].limit_sells/self.agents[self.execution_agent_id].initial_volume,                
                'time': t,
                'drift': mid_price - initial_mid_price,
                'n_events': self.agents['noise_agent'].n_events,
                'terminated': terminated, 
                }
        if self.execution_agent_id == 'rl_agent':
            # if rl agent is present, get an observation from the market 
            # this observation then feeds into the neural network 
            observation = self.agents[self.execution_agent_id].get_observation(t, self.lob)
        else:
            # benchmark agents return no observation 
            observation = np.array([None], dtype=np.float32)            
        return observation, transition_reward, terminated, info 

def make_env(config):
    def thunk():
        return Market(config)
    return thunk

def rollout_vectorized_rl(n_episodes, env_fns,
                          Model, model_path, device 
                          ):
    """        
        - loads rl policy and rolls it out in a vectorized manner
        - seed is used via seed+s (s in range(n_envs))

    note:
        - choosing cpu or cuda as device doesn't seem to make any difference 
        - we can max out the n_envs. for some reason one can even choose n_envs > n_proc without any errors                 
    """
    # load environment and model
    env = gym.vector.AsyncVectorEnv(env_fns)
    agent = Model(env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    # prepare rollout
    rewards = []
    times = []
    n_events = []
    # start sampling 
    observation, info = env.reset()
    while len(rewards) < n_episodes:
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation).to(device))
        observation, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    rewards.append(info['reward'])
    return rewards

def rollout_vectorized_benchmarks(seed, n_episodes, n_envs, execution_agent, market_type, volume): 
    """
        - uses vectorized environment to roll out benchmark rewards 
        - seed is used via seed+s (s in range(n_envs))

    note: 
        - mp_rollout is faster for benchmark agents but slower for the rl agent (maybe because of the forward pass in the neural network)        
        - not using this method 
    """
    assert execution_agent != 'rl_agent'    
    samples_per_env = int(n_episodes/n_envs) 
    configs = [{'seed': seed+s, 'market_env': market_type, 'execution_agent': execution_agent, 'volume': volume} for s in range(n_envs)]
    env_fns = [make_env(c) for c in configs]
    M = gym.vector.AsyncVectorEnv(env_fns) 
    total_rewards = []
    times = []
    n_events = []
    for _ in range(samples_per_env):
        _, info = M.reset()
        total_rewards.extend(list(info['reward']))
        times.append(list(info['time']))
        n_events.append(list(info['n_events']))
    return total_rewards, times, n_events

def rollout(seed, n_episodes, execution_agent, market_type, volume, terminal_time, time_delta):
    """
    rollout episodes for an execution agent

    Args:
        n_episodes: number of episodes to rollout
        execution_agent: the execution agent to use: sl, linear_sl, or rl 
        market_type: noise, flow, or strategic 
        volume : number of lots to trade. must be divisible by terminal_time/time_delta 
        terminal_time: the time at which the simulation should terminate
        time_delta: the time difference at which the agent trades 
    
    Returns:
        total_rewards: list of rewards for each episode
        times: list of terminal times for each episode
        n_events: list of number of events for each episode
    
    Raises:
        errors are raised within the Market Class, if the input configurations are wrong         
    """

    config = {'seed': seed, 'market_env': market_type, 'execution_agent': execution_agent, 'volume': volume, 'terminal_time': terminal_time, 'time_delta': time_delta}
    M = Market(config)
    total_rewards = []
    times = []
    n_events = []
    for _ in range(n_episodes):
        # M.reset() will run until the inventory is depletet for the benchmark agents 
        observation, info = M.reset()
        if execution_agent == 'rl_agent':
            # if an rl_agent is present, the environment runs until the next observation every time_delta 
            # it terminates if the execution agent's inventory is depleted
            print('NEW EPISODE')
            terminated = False
            while not terminated:
                # action depends on the dimension of the action space 
                action = np.array([0,0,1,0,0,0,0], dtype=np.float32)
                assert action in M.action_space
                observation, reward, terminated, truncated, info = M.step(action)
                assert observation in M.observation_space
        total_rewards.append(info['reward'])
        times.append(info['time'])
        n_events.append(info['n_events'])        
    return total_rewards, times, n_events

def mp_rollout(n_samples, n_cpus, execution_agent, market_type, volume, seed, terminal_time, time_delta):
    """
    rolls out episodes in parallel using multiprocessing

    Args:
        n_samples: number of episodes to rollout
        n_cpus: number of cpus to use
        execution_agent: the execution agent to use: sl, linear_sl, or rl
        market_type: noise, flow, or strategic
        volume : number of lots to trade. must be divisible by terminal_time/time_delta
        seed: seed for the random number generator
    
    Returns:
        all_rewards: list of rewards for each episode
        times: list of terminal times for each episode
        n_events: list of number of events for each episode

    Note: This is slightly faster than the vectorized rollout for the benchmark agents
    We mainly use this function to generate rollouts 
        
    """
    samples_per_env = int(n_samples/n_cpus) 
    print('')
    print(f'##### starting rollout #####')
    print(f'running {samples_per_env} episodes per CPU')
    print(f'using {n_cpus} CPUs')
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed+s, samples_per_env, execution_agent, market_type, volume, terminal_time, time_delta) for s in range(n_cpus)])    
    all_rewards, times, n_events  = zip(*out)
    all_rewards = list(itertools.chain.from_iterable(all_rewards))
    times = list(itertools.chain.from_iterable(times))
    n_events = list(itertools.chain.from_iterable(n_events))
    return all_rewards, times, n_events

if __name__ == '__main__':
    
    #### run full parallel benchmark rollouts for all environments, lot sizes, execution agents 
    saving_directory = 'rewards'
    # n_sample should be a multiple of n_cpus
    n_samples = int(1e4)
    # n_samples = 5000
    n_cpus = 128
    seed = 100
    envs = ['noise', 'flow', 'strategic']
    # envs = ['strategic']
    n_lots = [20, 60]
    # n_lots = [10, 60]    
    agents = ['sl_agent', 'linear_sl_agent']
    # agents = ['linear_sl_agent']

    for env in envs:
        for lots in n_lots:
            for agent in agents:
                print(f'env: {env}, lots: {lots}, agent: {agent}')
                start_time = time.time()
                rewards, times, n_events = mp_rollout(n_samples=n_samples, n_cpus=n_cpus, execution_agent=agent, market_type=env, volume=lots, seed=seed, terminal_time=150, time_delta=15)
                np.savez(f'rewards/{env}_{lots}_episodes_{n_samples}_eval_seed_{seed}_{agent}.npz', rewards=rewards)
                end_time = time.time()
                execution_time = end_time - start_time
                print("Execution time:", execution_time)
                print(f'mean rewards: {np.mean(rewards)}')
                print(f'length of rewards: {len(rewards)}')
    
    #### run rollouts  
    # n_samples = 3
    # # n_cpus = 5
    # # agent = 'linear_sl_agent'
    # agent = 'sl_agent'
    # # agent = 'rl_agent'
    # env = 'strategic'
    # # env = 'noice'
    # # env = 'flow'
    # lots = 10
    # seed = 100
    # # rollout 
    # start_time = time.time()
    # rewards, times, n_events = rollout(seed=0, n_episodes=n_samples, execution_agent=agent, market_type=env, volume=lots, terminal_time=150, time_delta=15)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # print(f'rewards: {rewards}')
    # print(f'times: {times}')

    # rollout benchmark with multiprocessing 
    # start_time = time.time()
    # rewards, times, n_events = mp_rollout(n_samples=n_samples, n_cpus=n_cpus, execution_agent=agent, market_type=env, volume=lots, seed=seed)
    # np.savez(f'raw_rewards/std4_rewards_{env}_{lots}_{agent}.npz', rewards=rewards)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # # print(rewards)
    # print(f'mean rewards: {np.mean(rewards)}')
    # print(f'length of rewards: {len(rewards)}')
    
    # start_time = time.time()
    # # this is only about 1 second slower than mp rollout for 100 samples and 80 cpus
    # total_rewards, times, n_events = rollout_vectorized_benchmarks(seed=0, n_episodes=n_samples, execution_agent='linear_sl_agent',  n_envs=n_cpus, market_type=env, volume=lots)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # # print(total_rewards)
    # print(f'mean rewards: {np.mean(total_rewards)}')
