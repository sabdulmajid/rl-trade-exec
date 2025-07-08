# computing heat map plots ! 
#%%
import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from limit_order_book.plotting import heat_map
import matplotlib.pyplot as plt
from simulation.market_gym import Market

# %%
seed = 2 

config = {'seed': seed, 'market_env': 'noise', 'execution_agent': 'linear_sl_agent', 'volume':int(11*5), 'terminal_time': 165, 'time_delta': 15}
M = Market(config)
total_rewards = []
times = []
n_events = []
print('starting')
observation, info = M.reset()
print(info)


level2, orders, market_orders = M.lob.log_to_df()
textwidth = 16
textheight = 9
scale=0.75
heat_map(trades=market_orders, level2=level2, event_times=level2.time, max_level=7, scale=1750, max_volume=40, width=scale*textwidth, height=scale*textheight)

plt.tight_layout()
plt.savefig(f'plots/heat_map_seed_{seed}.pdf')




# %%
