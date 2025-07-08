# Get the current script's directory
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)# Add the parent directory to sys.path
sys.path.append(parent_dir)
# import 
import numpy as np
# from advanced_multi_lot import Market 
import matplotlib.pyplot as plt
from matplotlib import cm 
import pandas as pd 

# TODO: Wrap all the data into one data frame 


def heat_map(trades, level2, event_times, max_level=30, scale=1000, max_volume=1000, xlim=[0,150], ylim=[995,1005], width=6.75, height=9):
    '''
    inputs:
        - trades: data frame with columns ['type', 'side', 'size', 'price']
        - level2: data frame with columns ['best_bid_price', 'best_ask_price', 'best_bid_volume', 'best_ask_volume', 'bid_price_0', 'bid_volume_0', 'ask_price_0', 'ask_volume_0', ...]

    output:
        - out: heatmap of the book 
    '''

    bid_prices = [f'bid_price_{n}' for n in range(max_level)] 
    bid_prices= np.hstack(np.array(level2[bid_prices]))
    ask_prices = [f'ask_price_{n}' for n in range(max_level)]
    ask_prices = np.hstack(np.array(level2[ask_prices]))
    bid_volumes = [f'bid_volume_{n}' for n in range(max_level)]
    bid_volumes = -1*np.hstack(np.array(level2[bid_volumes]))
    ask_volumes = [f'ask_volume_{n}' for n in range(max_level)]
    ask_volumes = np.hstack(np.array(level2[ask_volumes]))

    # comment this line to use use tick time: 1,2,3, and so on 
    trades.time = event_times
    time = np.array(event_times)
    N = len(time)

    prices = np.hstack([bid_prices, ask_prices])
    volumes = np.hstack([bid_volumes, ask_volumes])
    # time = np.arange(N)
    extended_time = []
    for n in range(N):
        extended_time.extend(max_level*[time[n]])
    for n in range(N):
        extended_time.extend(max_level*[time[n]])

    trades[['buy', 'sell']] = trades[['buy', 'sell']].shift(-1)
    # bid_mask = (trades.side == 'bid') & (trades.type == 'M')
    # ask_mask = (trades.side == 'ask') & (trades.type == 'M')     
    # max_volume = max(trades['size'][trades.type == 'M'])
    # hard coded. find better logic for this. 
    # max_volume = 1000

    # width, height 
    plt.figure(figsize=(width, height), dpi=300)

    plt.scatter(extended_time, prices, c=volumes, cmap=cm.bwr, vmin=-max_volume, vmax=max_volume, s=150, label='_nolegend_')
    # plt.scatter(extended_time, prices, c=volumes, cmap=cm.seismic, vmin=-max_volume, vmax=max_volume, s=150, label='_nolegend_', alpha=1.0)

    plt.plot(time, level2.best_bid_price, '-', color='black', linewidth=5, label='_nolegend_')
    plt.plot(time, level2.best_ask_price, '-', color='black', linewidth=5, label='Best bid/ask prices')
    cbar = plt.colorbar()
    cbar.set_label('Volume', rotation=270, labelpad=15, fontsize=22)

    M = trades[['buy', 'sell']].max().max()
    plt.scatter(trades[trades.buy>0].time.values, level2.best_ask_price[trades.buy>0], color='black', marker='^', s= (scale/M)*trades[trades.buy>0].buy.values, label='Market buy') 
    plt.scatter(trades[trades.sell>0].time.values, level2.best_bid_price[trades.sell>0], color='black', marker='v', s= (scale/M)*trades[trades.sell>0].sell.values, label='Market sell')
    
    plt.xlim(xlim[0], 150)
    plt.ylim(994.5,1003.5)
    plt.xticks(np.arange(0, 151, step=15))


    
    # handles, labels = plt.gca().get_legend_handles_labels()
    # Set x and y tick size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Time', fontsize=22)
    plt.ylabel('Price', fontsize=22)
    

    # lg = plt.legend(['best ask price', 'best bid price', 'market buy', 'market sell'], prop={'size': 12}, loc='upper left')
    lg = plt.legend(prop={'size': 16}, loc='upper left')
    # print(lg.legendHandles[2]._sizes)
    # print(lg.legendHandles[0]._sizes)
    lg.legendHandles[2]._sizes = [150]
    lg.legendHandles[1]._sizes = [150]
    # lg.legendHandles[3]._sizes = [150]
    

    return None  

def plot_average_book_shape(bid_volumes, ask_volumes, ax, level=3, symetric=False, file_name='shape', title='average shape'):
    """
    - bid/ask_volumes: list of np arrays, [v1, v2, v3, ...]
    """ 

    level = len(bid_volumes[0]) 
    # T = len(bid_volumes)
    # book_shape_bid = np.nanmean(bid_volumes[-int(T/2):][::100], axis=0)
    # book_shape_ask = np.nanmean(ask_volumes[-int(T/2):][::100], axis=0)

    book_shape_bid = np.nanmean(bid_volumes, axis=0)[:level]
    book_shape_ask = np.nanmean(ask_volumes, axis=0)[:level]


    # plt.figure(figsize=(10, 6))     
    ax.grid(zorder=0)
    if symetric:
        shape = (book_shape_bid + book_shape_ask)/2
        ax.bar(range(0,-level,-1), shape, color='blue', label='bid')
        ax.bar(range(1,level+1,1), shape, color='red', label='ask')    
    else:
        ax.bar(range(0,-level,-1), book_shape_bid, color='blue', label='bid')
        ax.bar(range(1,level+1,1), book_shape_ask, color='red', label='ask')
    # ax.legend(loc='upper right')
    ax.legend(loc='upper right', prop={'size': 16})
    # ax.set_xlabel('relative distance to mid price')
    # ax.set_xlabel('relative distance to mid price', fontsize=18)
    # ax.set_yticks(range(0, 26, 5))
    ax.set_ylabel('Volume', fontsize=16)
    ax.set_xlabel('Ticks', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlim(-level-1,level+1)
    # ticks at 1,10,..,level
    upper_xticks = list(range(0,level+1, 10))
    upper_xticks[0] = 1
    # ticks at 0, -9,..,-level+1
    lower_xticks = list(range(1,-level, -10))[1:]
    xticks = lower_xticks + upper_xticks
    ax.set_xticks(xticks)
    lower_label = list(range(-10,-level-1,-10))
    # label2[0] = 1 
    # xtick_labels = label1 + label2
    ax.set_xticklabels(lower_label + upper_xticks)
    # ax.set_title(title)
    # ax.set_ylim(0, 25)
    # ax.set_title(title, fontsize=18)
    # ax.set_title(title, fontsize=18)
    # ax.tight_layout()
    # ax.savefig(f'{file_name}.pdf')
    return None


    # TODO: analyze average book shape
    return None 

def plot_prices(level2, trades, marker_size=50):
    """
    the method plots 
        - bid and ask prices 
        - microprice 
        - trades on bid and ask (larger trade with larger marker size)            
    """

    level2 = level2.copy()

    level2['micro_price'] = (level2.best_bid_price * level2.best_ask_volume + level2.best_ask_price * level2.best_bid_volume) / (level2.best_bid_volume + level2.best_ask_volume)

    trades = trades.shift(-1)

    data = pd.concat([level2, trades], axis=1)

    bid_mask = (data.type == 'M') & (data.side == 'bid')
    ask_mask = (data.type == 'M') & (data.side == 'ask')
    max_volume = max(data['size'][data.type == 'M'])

    plt.figure(figsize=(10, 6))

    plt.plot(data.index, data.best_bid_price, '-', color='black')
    plt.plot(data.index, data.best_ask_price, '-', color='black')
    plt.plot(data.index, data.micro_price , '-', color='blue')


    plt.scatter(data.index[bid_mask], data.best_bid_price[bid_mask], color='black', marker='v', s= marker_size*data['size'][bid_mask]/max_volume)
    plt.scatter(data.index[ask_mask], data.best_ask_price[ask_mask], color='black', marker='^',s= marker_size*data['size'][ask_mask]/max_volume)

    fontsize = 18
    plt.xlim(0,2000)

    plt.ylim(996,1006)
    plt.legend(['best bid price', 'best ask price', 'micro price', 'market sell', 'market buy'], prop={'size': 16})
    
    plt.ylabel('Price', fontsize=fontsize)
    plt.xlabel('Simulation Steps', fontsize=fontsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    return None 




    


def plot_level2_order_book(bid_prices, ask_prices, bid_volumes, ask_volumes, n):
    """"
    input: 
        - bid/ask_prices: list of np arrays, [p1, p2, p3, ...]
        - bid/ask_volumes: list of np arrays, [v1, v2, v3, ...]
        - n: index of the order book snapshot
    output:
        - plot of the order book snapshot
    """
    plt.figure()
    plt.bar(bid_prices[n], bid_volumes[n], color='b')
    plt.bar(ask_prices[n], ask_volumes[n], color='r')
    return


if __name__ == '__main__': 
    config = {'total_n_steps': int(1e3), 'log': True, 'seed':0, 'initial_volume': 500, 'env_type': 'simple', 'ada':False}
    M = Market(config=config)
    print(f'initial volume is {config["initial_volume"]}')
    rewards = []
    for n in range(1):
        observation, _ = M.reset()
        assert observation in M.observation_space 
        terminated = truncated = False 
        reward_per_episode = 0 
        while not terminated and not truncated: 
            action = np.array([0, 1, 0, 0, 0], dtype=np.float32)
            assert action in M.action_space
            observation, reward, terminated, truncated, info = M.step(action, transform_action=False)
            assert observation in M.observation_space
            reward_per_episode += reward
        rewards.append(reward_per_episode)
        assert M.volume == 0 
    
    # ToDo: either make history all list or all np arrays   
    # Check logging mechanism in LOB 
    # plot_level2_order_book(M.bid_prices, M.ask_prices, M.bid_volumes, M.ask_volumes, 0)
    # plot_average_book_shape(M.bid_volumes, M.ask_volumes)
    # plot_prices(M.best_bid_prices, M.best_ask_prices, M.best_bid_volumes, M.best_ask_volumes, M.trades)
    plt.show()
    # plt.figure()
    heat_map(np.array(M.best_ask_prices), np.array(M.best_bid_prices), M.bid_prices, M.ask_prices, M.bid_volumes, M.ask_volumes, M.trades)
    plt.show()