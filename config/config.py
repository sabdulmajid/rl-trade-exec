import numpy as np
import pandas as pd 

# parameters from the paper 
limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
market_intensity = 0.1237
cancel_intensities = np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
base ={}
base['limit_intensities'] = limit_intensities
base['market_intensities'] = market_intensity
base['cancel_intensities'] = 1e-3*cancel_intensities
base['distribution'] = 'log_normal'
base['market_mean'] = 4
base['market_std'] = 1.19
base['limit_mean'] = 4.47
base['limit_std'] = 0.83
base['cancel_mean'] = 4.48
base['cancel_std'] = 0.82
base['volume_min'] = 1
base['volume_max'] = 2000

#
noise_agent_config = {}

# intensities
noise_agent_config['market_intensity'] = market_intensity
noise_agent_config['limit_intensities'] = limit_intensities
noise_agent_config['cancel_intensities'] = 1e-1*cancel_intensities

# volume related things 
noise_agent_config['volume_distribution'] = 'half_normal'
noise_agent_config['market_mean'] = 0
noise_agent_config['market_std'] = 2
noise_agent_config['limit_mean'] = 0
noise_agent_config['limit_std'] = 2
noise_agent_config['cancel_mean'] = 0
noise_agent_config['cancel_std'] = 2
noise_agent_config['volume_min'] = 1
noise_agent_config['volume_max'] = 20
noise_agent_config['unit_volume'] = False
noise_agent_config['level'] = 30 
noise_agent_config['fall_back_volume'] = 5

# noise agent config 
noise_agent_config['initial_shape'] = None
noise_agent_config['initial_shape_file'] = None 
noise_agent_config['damping_factor'] = 0.65
noise_agent_config['imbalance_reaction'] = False
noise_agent_config['imbalance_factor'] = 2.0
noise_agent_config['default_waiting_time'] = 1e-6
noise_agent_config['intensity_scaling'] = 0.85

# 
noise_agent_config['rng'] = np.random.default_rng(0)
# 
noise_agent_config['initial_bid'] = 1000
noise_agent_config['initial_ask'] = 1001

# 
noise_agent_config['start_time'] = -15
noise_agent_config['terminal_time'] = None
noise_agent_config['priority'] = 1

# sl 
sl_agent_config = {}
sl_agent_config['volume'] = None
sl_agent_config['terminal_time'] = None
sl_agent_config['start_time'] = 0 
sl_agent_config['priority'] = 0

# linear sl 
linear_sl_agent_config = {}
linear_sl_agent_config['volume'] = None
linear_sl_agent_config['terminal_time'] = None
linear_sl_agent_config['start_time'] = 0
linear_sl_agent_config['time_delta'] = 500
linear_sl_agent_config['priority'] = 0

# market 
market_agent_config = {}
market_agent_config['volume'] = None
market_agent_config['start_time'] = 0
market_agent_config['priority'] = 0

# rl
rl_agent_config = {}
rl_agent_config['volume'] = None
rl_agent_config['terminal_time'] = None
rl_agent_config['start_time'] = 0
rl_agent_config['time_delta'] = 500
rl_agent_config['priority'] = 0
rl_agent_config['action_book_levels'] = 5 
rl_agent_config['observation_book_levels'] = 5
rl_agent_config['initial_shape_file'] = None 

# strategic
strategic_agent_config = {}
strategic_agent_config['start_time'] = -15
strategic_agent_config['time_delta'] = 3
strategic_agent_config['market_volume'] = 1
strategic_agent_config['limit_volume'] = 2
strategic_agent_config['rng'] = None 
strategic_agent_config['priority'] = 2
strategic_agent_config['terminal_time'] = 150

# initial 
initial_agent_config = {}   
initial_agent_config['start_time'] = -15
initial_agent_config['initial_bid'] = 1000
initial_agent_config['initial_ask'] = 1001
initial_agent_config['n_initial_levels'] = 30
initial_agent_config['initial_shape'] = None 
initial_agent_config['initial_shape_file'] = None
initial_agent_config['priority'] = -2

# 
observation_agent_config = {}
observation_agent_config['priority'] = -1
observation_agent_config['agent_id'] = 'observation_agent'
observation_agent_config['start_time'] = 0
observation_agent_config['terminal_time'] = 135
observation_agent_config['time_delta'] = 15

