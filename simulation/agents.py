# this file contains all agents relevant for the simulation 

import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
from limit_order_book.limit_order_book import LimitOrder, MarketOrder, CancellationByPriceVolume, Cancellation
from config.config import noise_agent_config
# for convolution 
import torch 
import scipy 
# from numba import njit
import bisect 

# currently not used 
# @njit
# def convolve_columns_numba(arr, filter_kernel, stride=1):
#     num_rows, num_cols = arr.shape
#     filter_size = len(filter_kernel)
#     output_size = (num_rows - filter_size) // stride + 1
    
#     convolved = np.empty((output_size, num_cols))
    
#     for col in range(num_cols):
#         for out_row in range(output_size):
#             acc = 0.0
#             for k in range(filter_size):
#                 acc += arr[out_row * stride + k, col] * filter_kernel[k]
#             convolved[out_row, col] = acc
    
#     return convolved

class NoiseAgent(): 
    def __init__(self, rng, 
                 level,
                 limit_intensities, market_intensity, cancel_intensities,
                 intensity_scaling, 
                 initial_shape_file, 
                 initial_shape, 
                 volume_distribution,
                 market_mean, market_std,
                 cancel_mean, cancel_std,
                 limit_mean, limit_std,
                 volume_min, volume_max,
                 unit_volume,
                 damping_factor, imbalance_reaction, imbalance_factor,                 
                 initial_bid, 
                 initial_ask, 
                 start_time,
                 terminal_time,
                 priority,
                 fall_back_volume, 
                 default_waiting_time=1e-6
                 ):
        """"    
        Parameters:
        ----
        initial_shape_file: str
            stationary shape of the order book to start from 
            if this is None, use the integer stored in initial_shape 
        level: int 
            number of levels the noise agents placed its orders. also the number of levels to initialize the order book.
            numbers of orders outside this range are cancelled
        limit_intensities, market_intensity, cancel_intensities: np.array of intensities        
        imbalance_reaction: bool 
            if True, the agent reacts to the imbalance of the order book 
        damping_factor: damping factor for the imbalance computation 
        volume_distribtion: half normal or log normal 
        rng: np.random number generator instance                                 
        """
        
        # this is used to scale down intensities 
        self.intensity_scaling = intensity_scaling

        self.damping_factor = damping_factor        
        self.damping_weights = np.exp(-self.damping_factor*np.arange(level)) # move damping weights here for speed up
        # controls whether imbalance calculation starts at the best prices 
        self.start_imbalance_at_best = True
        self.imbalance_reaction = imbalance_reaction
        self.imbalance_factor = imbalance_factor

        self.np_random = rng 
        self.level = level 
        self.initial_level = self.level # number of levels to initialize (could be bigger than self.level)
        assert self.initial_level >= self.level, 'initial level must be bigger than level'
        self.initial_bid = initial_bid
        self.initial_ask = initial_ask
                
        assert self.level >= 10, 'level must be at least 10'
        self.limit_intensities = np.pad(limit_intensities, (0,self.level-len(limit_intensities)), 'constant', constant_values=(0))
        self.cancel_intensities = np.pad(cancel_intensities, (0,self.level-len(cancel_intensities)), 'constant', constant_values=(0))
        self.market_intensity = market_intensity

        # volume distribution. could seperate this out into another class 
        self.unit_volume = unit_volume
        self.volume_distribution = volume_distribution
        self.market_mean = market_mean
        self.market_std = market_std
        self.limit_mean = limit_mean
        self.limit_std = limit_std
        self.cancel_mean = cancel_mean
        self.cancel_std = cancel_std
        self.volume_min = volume_min
        self.volume_max = volume_max
        
        # we use this if both sides become empty 
        self.fall_back_volume = fall_back_volume

        self.agent_id = 'noise_agent'

        self.current_order = None 
        self.waiting_time = None 
        self.default_waiting_time = default_waiting_time
        self.start_time = start_time
        self.terminal_time = terminal_time

        self.priority = priority

        self.n_events = None 

        return None  
    
    def reset_random_seet(self, rng):      
        self.np_random = rng
        return None
    
    # def initialize(self, time): 
        # ToDo: initial bid and ask as variable 
        # orders = [] 
        # for idx, price in enumerate(np.arange(self.initial_ask-1, self.initial_ask-1-self.initial_level, -1)):
        #     order = LimitOrder(agent_id=self.agent_id, side='bid', price=price, volume=self.initial_shape[idx], time=time)
        #     orders.append(order)
        # for idx, price in enumerate(np.arange(self.initial_bid+1, self.initial_bid+self.initial_level+1, 1)): 
        #     order = LimitOrder(agent_id=self.agent_id, side='ask', price=price, volume=self.initial_shape[idx], time=time)
        #     orders.append(order)
        # return orders

    def volume(self, action):
        if self.unit_volume:
            return 1        
        assert self.volume_distribution in ['log_normal', 'half_normal'], 'distribution not implemented'
        # ToDo: initialize distribution at the beginning 
        if self.volume_distribution == 'log_normal':
            if action == 'limit':
                volume = self.np_random.lognormal(self.limit_volume_parameters['mean'], self.limit_volume_parameters['std'])   
            elif action == 'market':
                volume = self.np_random.lognormal(self.market_volume_parameters['mean'], self.market_volume_parameters['std'])
            elif action == 'cancellation':
                volume = self.np_random.lognormal(self.cancel_volume_parameters['mean'], self.cancel_volume_parameters['std'])
            volume = np.rint(np.clip(1+np.abs(volume), self.volume_min, self.volume_max))

        else: 
            if action == 'limit':
                volume = self.np_random.normal(self.limit_mean, self.limit_std)   
            elif action == 'market':
                volume = self.np_random.normal(self.market_mean, self.market_std)
            elif action == 'cancellation':
                volume = self.np_random.normal(self.cancel_mean, self.cancel_std)
            volume = np.rint(np.clip(1+np.abs(volume), self.volume_min, self.volume_max))

        return volume

    def generate_order(self, lob, time):
        """
        This methods samples an order and a time to execute this order 

        Parameters:
            - the current order book: it is expected to have the attributes best_bid_price, best_ask_price, bid_volumes, ask_volumes   
            - the current time 

        Returns:
            - order: class instance of LimitOrder, MarketOrder or CancellationByPriceVolume
            - time: float item time 

        """
        assert time >= lob.time 
        assert time >= self.start_time

        # if time == self.start_time:
        #     self.waiting_time = 0 
        #     return self.initialize(time)
        
        best_bid_price = lob.data.best_bid_prices[-1]
        best_ask_price = lob.data.best_ask_prices[-1]
        bid_volumes = lob.data.bid_volumes[-1]
        ask_volumes = lob.data.ask_volumes[-1]

        # handling of nan best bid price 
        if np.isnan(best_bid_price):
            # volume = self.initial_shape[0]
            # if volume is None:
            #     raise ValueError('volume is None')
            if np.isnan(best_ask_price):
                # bid and ask are nan 
                order = LimitOrder(agent_id=self.agent_id, side='bid', price=self.initial_bid, volume=self.fall_back_volume, time=time)
                print('both bid and ask are nan')
            else:
                # bid is nan, ask is not nan
                volume = lob.price_volume_map['ask'][best_ask_price]
                order = LimitOrder(agent_id=self.agent_id, side='bid', price=best_ask_price-1, volume=volume, time=time)
                print('bid is nan')
            self.waiting_time = self.default_waiting_time
            # Assign a value to self.waiting_time
            # if order.volume is None:
            #     raise ValueError('order is None')
            return [order]
        if np.isnan(best_ask_price):
            # ask is nan, bid is not nan 
            # volume = self.initial_shape[0]
            # if volume is None:
            #     raise ValueError(f'volume is None, initial shape is {self.initial_shape[0]}')

            volume = lob.price_volume_map['bid'][best_bid_price]
            # assert volume is not None 
            order = LimitOrder(agent_id=self.agent_id, side='ask', price=best_bid_price+1, volume=volume, time=time)
            print('ask is nan')
            # TODO: create better logic here 
            self.waiting_time = self.default_waiting_time
            return [order]

        assert len(bid_volumes) == len(ask_volumes), 'bid and ask volumes must have the same length'
        assert np.all(bid_volumes >= 0), 'All entries of bid volumes must be >= 0'
        assert np.all(ask_volumes >= 0), 'All entries of ask volumes must be >= 0'

        bid_cancel_intensity = self.cancel_intensities*bid_volumes 
        ask_cancel_intensity = self.cancel_intensities*ask_volumes
        if self.imbalance_reaction:
            L = len(bid_volumes)
            if np.sum(bid_volumes) == 0:
                weighted_bid_volumes = 0
            else:
                if self.start_imbalance_at_best:
                    spread = int(best_ask_price - best_bid_price)
                    # idx = np.nonzero(bid_volumes)[0][0]
                    # assert spread - 1  == idx 
                    # L = 30, spread = 2, 0:L-1, 1:L 
                    weighted_bid_volumes = np.sum(self.damping_weights[:L-spread+1]*bid_volumes[spread-1:])
                else:
                    weighted_bid_volumes = np.sum(self.damping_weights*bid_volumes)
            if np.sum(ask_volumes) == 0:
                weighted_ask_volumes = 0
            else:
                if self.start_imbalance_at_best:
                    spread = int(best_ask_price - best_bid_price)
                    # idx = np.nonzero(ask_volumes)[0][0]
                    # assert spread - 1  == idx 
                    weighted_ask_volumes = np.sum(self.damping_weights[:L-spread+1]*ask_volumes[spread-1:])
                # idx = np.nonzero(ask_volumes)[0][0]
                # weighted_ask_volumes = np.sum(self.damping_weights[:L-idx]*ask_volumes[idx:])
                else:
                    weighted_ask_volumes = np.sum(self.damping_weights*ask_volumes)
            if (weighted_bid_volumes + weighted_ask_volumes) == 0:
                imbalance = 0
            else:
                imbalance = (weighted_bid_volumes - weighted_ask_volumes) / (weighted_bid_volumes + weighted_ask_volumes)
            if np.isnan(imbalance):
                print(imbalance)
                print(bid_volumes)
                print(ask_volumes)
                raise ValueError('imbalance is nan')
            assert -1 <= imbalance <= 1, 'imbalance must be in [-1, 1]'
            pos = self.imbalance_factor*max(0, imbalance)
            neg = self.imbalance_factor*max(0, -imbalance)
            # pos = max(0, imbalance)
            # neg = max(0, -imbalance)
            # possible adjustments of the imbalance: 
            # imbalance = np.sign(imbalance)*np.power(np.abs(imbalance), 1/2)
            # imbalance = np.sign(imbalance)
            # if I = 1, price goes up --> more market buys
            # market_buy_intensity = self.market_intesity*(1+imbalance)
            # market_sell_intensity = self.market_intesity*(1-imbalance)
            market_buy_intensity = self.market_intensity*(1+pos)
            market_sell_intensity = self.market_intensity*(1+neg)
            # adjust cancellation intensities
            # if imbalance = 1, price goes up --> cancel limit sell orders (ask)
            # adjust cancellation intensities
            # if imbalance = 1, price goes up --> cancel limit buy orders (bid)
            #
            # c = 0.5
            bid_cancel_intensity = bid_cancel_intensity*(1+neg)
            ask_cancel_intensity = ask_cancel_intensity*(1+pos)
            # n = 30
            # bid_cancel_intensity = bid_cancel_intensity*(1-imbalance)
            # ask_cancel_intensity = ask_cancel_intensity*(1+imbalance)
            # bid_cancel_intensity[:n] = bid_cancel_intensity[:n]*(1-imbalance*self.damping_weights[:n])
            # ask_cancel_intensity[:n] = ask_cancel_intensity[:n]*(1+imbalance*self.damping_weights[:n])
            # weights = np.exp(-0.1*np.arange(L))
            # bid_cancel_intensity = bid_cancel_intensity*(1-imbalance*(2*weights-1))
            # ask_cancel_intensity = ask_cancel_intensity*(1+imbalance*(2*weights-1))
            # imbalance*bid_limit_intensities[:n]*self.damping_weights[:n]            
            # adjust limit order intensities
            # if I=1, price goes up --> more limit buy orders
            ##
            # c = 0.5
            bid_limit_intensities = self.limit_intensities.copy()
            ask_limit_intensities = self.limit_intensities.copy()
            bid_limit_intensities = bid_limit_intensities*(1+pos)
            ask_limit_intensities = ask_limit_intensities*(1+neg)
            # bid_limit_intensities = bid_limit_intensities*(1+imbalance)
            # ask_limit_intensities = ask_limit_intensities*(1-imbalance)
            # bid_limit_intensities = bid_limit_intensities*(1+imbalance*(2*weights-1))
            # ask_limit_intensities = ask_limit_intensities*(1-imbalance*(2*weights-1))
            # imbalance*bid_limit_intensities[:n]*self.damping_weights[:n]
            # bid_limit_intensities[:n] = bid_limit_intensities[:n]*(1+0.001*pos)        
        else:
            market_buy_intensity = market_sell_intensity = self.market_intensity
            bid_limit_intensities = ask_limit_intensities = self.limit_intensities

        # TODO: write all of this into a seperate function which outputs the updated intensities 
        probability = np.array([market_sell_intensity, market_buy_intensity, np.sum(bid_limit_intensities), np.sum(ask_limit_intensities), np.sum(bid_cancel_intensity), np.sum(ask_cancel_intensity)])        
        assert np.all(probability >= 0), 'all probabilities must be > 0'
        waiting_time = self.np_random.exponential(1/np.sum(probability))
        # Note: we save this into an attribute such that it doesnt need to be recomputet later 
        self.waiting_time = waiting_time
        # waiting_time = 1 

        probability = probability/np.sum(probability)
        assert np.abs(np.sum(probability)-1) < 1e-6, f'{probability} doesnt sum to one'
        action, side = self.np_random.choice([('market', 'bid'), ('market', 'ask'), ('limit', 'bid'), ('limit', 'ask'), ('cancellation', 'bid'), ('cancellation', 'ask')], p=probability)

        volume = self.volume(action)

        if action == 'limit': 
            if side == 'bid': 
                probability = bid_limit_intensities/np.sum(bid_limit_intensities)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_ask_price - level
            else: 
                probability = ask_limit_intensities/np.sum(ask_limit_intensities)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_bid_price + level
            order = LimitOrder(agent_id=self.agent_id, side=side, price=price, volume=volume, time=time) 

        elif action == 'market':
            order = MarketOrder(agent_id=self.agent_id, side=side, volume=volume, time=time)
        
        elif action == 'cancellation':
            if side == 'bid':
                probability = bid_cancel_intensity/np.sum(bid_cancel_intensity)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_ask_price - level
            elif side == 'ask':
                probability = ask_cancel_intensity/np.sum(ask_cancel_intensity)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_bid_price + level
            order = CancellationByPriceVolume(agent_id=self.agent_id, side=side, price=price, volume=volume, time=time)

        if order.volume is None:
            raise ValueError('order is None')

        return [order]

    def cancel_far_out_orders(self, lob, time):        
        # ToDo: Could add this as a function to the order book (as an order type)
        order_list = []
        for price in lob.price_map['bid'].keys():
            if price < lob.get_best_price('ask') - self.level:
                for order_id in lob.price_map['bid'][price]:
                    # this is repetitive 
                    if lob.order_map[order_id].agent_id == self.agent_id:
                        order = Cancellation(agent_id=self.agent_id, order_id=order_id, time=time)
                        order_list.append(order)
        # try to establish boundary conditions 
        # order = LimitOrder(agent_id=self.agent_id, side='bid', price=lob.get_best_price('ask') - self.level - 1, volume=self.initial_shape[0])
        # order_list.append(order)

        for price in lob.price_map['ask'].keys():
            if price > lob.get_best_price('bid') + self.level:
                for order_id in lob.price_map['ask'][price]:
                    # also repetitive 
                    if lob.order_map[order_id].agent_id == self.agent_id:
                        order = Cancellation(agent_id=self.agent_id, order_id=order_id, time=time)
                        order_list.append(order)                                
        # order = LimitOrder(agent_id=self.agent_id, side='ask', price=lob.get_best_price('bid') + self.level + 1, volume=self.initial_shape[0])
        # order_list.append(order)

        return order_list

    def new_event(self, time, event):
        assert self.waiting_time is not None 
        assert event == self.agent_id
        assert time >= self.start_time
        if time >= self.terminal_time:
            return None 
        waiting_time = self.waiting_time
        # just for safety reasons. to ensure that waiting time is set again
        self.waiting_time = None
        self.n_events += 1
        return (time+waiting_time, self.priority, self.agent_id)
    
    def initial_event(self):
        # TODO: add priority for agents 
        # assert self.waiting_time is None 
        # this sets self.waiting_time 
        # self.generate_order(LOB, self.start_time)
        # t = self.waiting_time
        # self.waiting_time = None
        # strictly speaking the start time should be self.start_time+exponential waiting time
        # but this is a bit annoying to implement 
        # the below should not make much of a difference 
        return (self.start_time, self.priority, self.agent_id)

    def reset(self):
        self.n_events = 0
        pass 

class ExecutionAgent():
    """
    Base class for execution agents.
        - keeps track of volume, active_volume, cummulative_reward, passive_fills, market_fills
        - active volume is the volume currently placed in the book 
        - update positin takes a message and updates volumes, passive fille, market fills, and rewards  
        - reset function 
    """
    def __init__(self, volume, agent_id, priority=0) -> None:
        self.initial_volume = volume
        self.agent_id = agent_id
        self.priority = priority
        self.reset()
    
    def reset(self):
        self.volume = self.initial_volume
        self.active_volume = 0
        self.cummulative_reward = 0
        self.reference_bid_price = None
        self.market_buys = 0 
        self.market_sells = 0
        self.limit_buys = 0
        self.limit_sells = 0

    def get_reward(self, cash, volume):        
        assert self.reference_bid_price is not None, 'reference bid price is not set'
        return (cash - volume * self.reference_bid_price) / self.initial_volume
    
    def update_position_from_message_list(self, message_list):
        rewards = [self.update_position(m) for m in message_list]                
        assert self.volume >= 0         
        terminated = self.volume == 0
        return sum(rewards), terminated

    def update_position(self, fill_message):
        reward = 0 
        assert self.active_volume >= 0
        assert self.volume >= 0
        assert self.limit_buys >= 0
        assert self.limit_sells >= 0
        assert self.market_buys >= 0
        assert self.market_sells >= 0
        assert self.active_volume <= self.volume
        assert self.market_buys + self.market_sells + self.limit_buys + self.limit_sells <= self.initial_volume
        if fill_message.type == 'modification':
            # this agent doesnt modify orders
            print('MODIFICATION but we do not update the position!')
            pass
        elif fill_message.type == 'cancellation_by_price_volume':
            # Note: cancellation by price and volume could also return a list of cancellations and modifications
            if fill_message.order.agent_id == self.agent_id:
                self.active_volume -= fill_message.filled_volume
            else:
                pass
        elif fill_message.type == 'limit':
            if fill_message.agent_id == self.agent_id:
                # this means the limit order was send into the book by the agent 
                self.active_volume += fill_message.volume
        elif fill_message.type == 'cancellation':
            if fill_message.order.agent_id == self.agent_id:
                self.active_volume -= fill_message.volume
        elif fill_message.type == 'market':
            # check for active market trades             
            side = fill_message.order.side
            if fill_message.order.agent_id == self.agent_id:
                assert side == 'bid', 'this execution agent only sells'
                # ask means buy --> volume increases, negative cash flow 
                if side == 'ask':
                    self.volume += fill_message.filled_volume
                    self.market_buys += fill_message.filled_volume
                    reward -= self.get_reward(fill_message.execution_price, fill_message.filled_volume)
                    self.cummulative_reward -= reward
                # bid means sell --> volume decreases, positive cash flow
                elif side == 'bid':
                    self.volume -= fill_message.filled_volume
                    self.market_sells += fill_message.filled_volume
                    reward += self.get_reward(fill_message.execution_price, fill_message.filled_volume)
                    self.cummulative_reward += reward
            # check for passive limit order fills 
            if self.agent_id in fill_message.passive_fills:
                assert side == 'ask', 'this execution agent only sells'
                cash = 0
                volume = 0 
                for m in fill_message.passive_fills[self.agent_id]:
                    volume += m.filled_volume
                    cash += m.filled_volume * m.order.price 
                # market side is ask, market buy --> limit sell 
                if side == 'ask':
                    self.active_volume -= volume
                    self.volume -= volume
                    self.limit_sells += volume
                    reward += self.get_reward(cash, volume)
                    self.cummulative_reward += reward
                # market side is bid, market sell --> limit buy
                elif side == 'bid':
                    self.active_volume -= volume
                    self.volume += volume
                    self.limit_buys += volume
                    reward -= self.get_reward(cash, volume)
                    self.cummulative_reward -= reward
            return reward
        else: 
            raise ValueError(f'Unknown message type {fill_message.type}')
        return 0
    
    def sell_remaining_position(self, lob, time):
        assert self.agent_id, 'agent id is not set' 
        assert self.volume > 0, 'volume is 0'
        # assert lob.order_map_by_agent[self.agent_id], 'agent has no orders in the book'
        # assert self.agent_id in lob.order_map_by_agent, 'agent has no orders in the book'
        order_list = []
        for order_id in lob.order_map_by_agent[self.agent_id]:
            order_list.append(Cancellation(agent_id=self.agent_id, order_id=order_id, time=time))
        order_list.append(MarketOrder(agent_id=self.agent_id, side='bid', volume=self.volume, time=time))
        return order_list

class MarketAgent(ExecutionAgent):

    def __init__(self, volume, start_time, priority) -> None:
        super().__init__(volume, 'market_agent', priority)
        self.start_time = start_time
                    
    def generate_order(self, lob, time):
        assert self.active_volume >= 0 
        assert self.volume >= 0
        if time == self.start_time:
            self.reference_bid_price = lob.get_best_price('bid')
            return [MarketOrder(self.agent_id, side='bid', volume=self.volume, time=time)]
        else:
            return None
    
    def get_observation(self, time, lob):
        return None 
    
    def new_event(self, time, event):
        # should never happen, raise error if it does 
        # TODO: improve logic here !! 
        raise ValueError('market agent only acts once')

    def initial_event(self):
        return (self.start_time, 0, self.agent_id)

class SubmitAndLeaveAgent(ExecutionAgent):

    def __init__(self, volume, start_time, terminal_time, priority) -> None:
        super().__init__(volume, 'sl_agent', priority)
        self.terminal_time = terminal_time
        self.start_time = start_time
        self.place_at_best_ask = True
                        
    def generate_order(self, lob, time):
        assert self.volume >= 0
        assert self.active_volume >= 0 
        assert 0 <= time <= self.terminal_time        
        if time == self.start_time:
            self.reference_bid_price = lob.get_best_price('bid')
            if self.place_at_best_ask:
                limit_price = lob.get_best_price('ask')
            else:
                limit_price = lob.get_best_price('bid')+1
            # print(f'reference bid price is {self.reference_bid_price}')
            # print(f'placing limit order at {limit_price}')            
            return [LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.initial_volume, time=time)]
        elif time == self.terminal_time:
            return self.sell_remaining_position(lob, time)
        else:
            raise ValueError('input time in execution agent order generation is wrong')
    
    def get_observation(self, time, lob):
        return None 
    
    def new_event(self, time, event):
        assert event == self.agent_id
        assert time == self.start_time, f'time : {time}, start time : {self.start_time}, event : {event}'
        assert self.volume > 0 
        # TODO: add logic for the terminal time step !! 
        return (self.terminal_time, self.priority, self.agent_id)
    
    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)
    
class LinearSubmitLeaveAgent(ExecutionAgent):

    def __init__(self, volume, start_time, time_delta, terminal_time, priority) -> None:
        super().__init__(volume=volume, agent_id='linear_sl_agent', priority=priority)
        assert start_time < terminal_time
        assert time_delta < terminal_time, 'time delta must be less than terminal time'
        assert terminal_time % time_delta == 0, 'terminal time must be divisible by time delta'
        time_steps = (terminal_time-start_time)/time_delta
        assert 0 < volume < time_steps or volume % time_steps == 0, 'volume must be divisible by time delta or volume < time delta'
        self.terminal_time = terminal_time
        self.start_time = start_time
        self.time_delta = time_delta
        self.volume_slice = volume/time_steps
        self.place_at_best_ask = True
        # self.action_times = np.arange(start_time, terminal_time+time_delta, time_delta)
        if volume >= time_steps:
            self.volume_slice = int(volume/time_steps)
            assert self.volume_slice * time_steps == volume
            self.submit_and_leave = False
        else:
            self.submit_and_leave = True
        return None 
                        
    def generate_order(self, lob, time):
        assert self.volume > 0 
        assert time >= self.start_time
        # (205-5)%100 == 0 
        if (time-self.start_time)%self.time_delta == 0:
            if time == self.start_time:
                self.reference_bid_price = lob.get_best_price('bid')
                best_ask = lob.get_best_price('ask')
                if best_ask is np.nan or not self.place_at_best_ask:
                    # either best_ask is np.nan or we don't place at the best ask 
                    limit_price = lob.get_best_price('bid')+1
                else:
                    limit_price = lob.get_best_price('ask')
                # if self.place_at_best_ask:
                #     limit_price = lob.get_best_price('ask')
                #     if limit_price is np.nan:
                #         limit_price = lob.get_best_price('bid')+1
                #     else:
                #         limit_price = lob.get_best_price('ask')
                # else:
                #     limit_price = lob.get_best_price('bid')+1
                if self.submit_and_leave:
                    return [LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.initial_volume, time=time)]
                else:
                    return [LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.volume_slice, time=time)]
            elif time < self.terminal_time:
                if self.submit_and_leave:
                    pass
                else:
                    limit_price = lob.get_best_price('bid')+1
                    return [LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.volume_slice, time=time)]
            elif time == self.terminal_time:
                return self.sell_remaining_position(lob, time)
            else:
                raise ValueError('time is in execution agent order generation is wrong')
        else:
            # in the current set up, this should not happen, because agent should only be triggered at the right times 
            # could also raise an error here 
            return None
    
    def get_observation(self, time, lob):
        return None 
    
    def new_event(self, time, event):
        assert event == self.agent_id
        assert event == self.agent_id
        assert time <= self.terminal_time
        assert time >= self.start_time
        assert (time - self.start_time) % self.time_delta == 0, 'time-start_time must be divisible by time delta'
        assert self.volume > 0 
        # we just took an action, what is the next event ? 
        # (time, priority, task)
        if self.submit_and_leave:
            return (self.terminal_time, self.priority, self.agent_id)
        else:
            return (time+self.time_delta, self.priority, self.agent_id)
            
    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)

class RLAgent(ExecutionAgent):
    """
        - this agent takes in an action and then generates an order

        attributes: 
            - ... 
        
        methods:
            - ... 
    """
    def __init__(self, action_book_levels, observation_book_levels, volume, terminal_time, start_time, time_delta, priority, initial_shape_file=None) -> None:
        """
        args:
            - action_book_levels: for example if this is = 2, post on the first two levels of the order book, action space is then [m,l1,l2,inactive]
            - observation_book_levels: for example if this is = 2, observe the first two levels of the order book
            - volume: volume to be traded
            - terminal_time: terminal time of the agent
            - start_time: start time of the agent
            - time_delta: time delta between actions 
            - priority: priority of the agent
        """
        assert action_book_levels >= 1, 'action book levels must be at least 1'
        super().__init__(volume, 'rl_agent', priority)
        self.orders_within_range = set()
        self.start_time = start_time
        self.terminal_time = terminal_time
        self.time_delta = time_delta
        # observation space length depends on features and queue positions, see the code below 
        # note: adding queue positions did not improve the result in the flow 40 case 
        # currently no better idea than setting this manually 
        # self.observation_space_length = 176
        # self.observation_space_length = 146 # shape with new queue position encoding 
        self.observation_space_length = 6+4+6+5+5+2*int(volume)
        # final shape is 6+4+6+5+5+60+60 with observable levels = 5 
        # market, l1, l2, inactive
        self.action_space_length = action_book_levels + 2 
        self.action_book_levels = action_book_levels
        self.observation_book_levels = observation_book_levels
        self.start_at_best_price = True

        if initial_shape_file is not None:
            self.initial_shape = (np.load(initial_shape_file)['bidv']+np.load(initial_shape_file)['askv'])/2
        else:
            # some normalization constant 
            self.initial_shape = 20
            
        
    def generate_order(self, lob, time, action):
        """ 
            input: 
                - action: 
                    - action is np.array 
                    - for example if action is np.array of length 6:  a[0] = market order, a[1] = limit first level, a[2] = limit second level, a[3] = limit third level, a[4] = limit forth level, a[5] = inactive orders
                    - if action is not in the simplex, we normalize it, such that its part of the simplex (at the moment we always normalize)                 
                - lob: current statte of the order book 
                - time: current time 

            returns: 
                - order_list: list of orders to be placed

        """
        assert self.start_time <= time <= self.terminal_time
        assert (time-self.start_time) % self.time_delta == 0, 'time must be divisible by time delta'
        assert len(action) == self.action_space_length, 'action length must be equal to action space length'

        if time >= self.start_time and time < self.terminal_time:            
            # normalize action such that its part of the simplex 
            # action = np.exp(action)/np.sum(np.exp(action), axis=0)
            assert np.all(action >= 0), 'all action values must be >= 0'
            assert np.abs(np.sum(action)-1) < 1e-6, 'action must sum to 1'
            best_bid = lob.get_best_price('bid')
            
            # target volumes: 
            # translate percentages to absolute value  
            # [market%, l1% ,l2%, ,l3%, l4%, inactive%] to [market, l1, l2, l3, l4, inactive]
            target_volumes = []
            available_volume = self.volume 
            for l in range(self.action_space_length):
                # np.round rounds values in [0, 0.5] to 0, and values in [0.5, 1] to 1 
                volume_on_level = min(np.round(action[l]*self.volume).astype(int), available_volume)
                assert volume_on_level >= 0
                available_volume -= volume_on_level
                target_volumes.append(volume_on_level)                
            # add any leftovers to the inactive level 
            target_volumes[-1] += available_volume
            assert sum(target_volumes) == self.volume

            # get agent order allocation 
            # if self.action_book_levels == 2, then volume_per_level = [l1, l2, l>2]
            volume_per_level, orders_within_range = self.get_order_allocation(lob, self.action_book_levels)            
            if self.action_book_levels >= 1:
                assert len(volume_per_level) + 1 == len(action)
            
            # cancel orders above best_bid+self.action_book_levels
            cancelled_volume = 0
            order_list = []
            orders_to_cancel = lob.order_map_by_agent[self.agent_id].difference(orders_within_range)
            for order_id in orders_to_cancel:
                order_list.append(Cancellation(self.agent_id, order_id, time))
                cancelled_volume += lob.order_map[order_id].volume            
            # last entry of self.volume_per_level contains volume on levels >= best_bid+self.action_book_levels
            assert cancelled_volume == volume_per_level[-1]
            
            # add 0 for the market order, just so the index is aligned 
            # volume per level is now [0, l1, l2, l3, l4, inactive]
            # target volumes: market, l1, l2, ..., ln, inactive 
            # volumes_per_level: 0, l1, l2, l3, ..., ln, l_>=n+1
            volume_per_level.insert(0, 0)
            l = 0 
            m = 0 
            for level in range(len(action)-1): 
                if level == 0:
                    if target_volumes[level] > 0:
                        order = MarketOrder(self.agent_id, 'bid', target_volumes[level], time)
                        order_list.append(order)
                        m = target_volumes[level]
                else:
                    diff = target_volumes[level] - volume_per_level[level]
                    limit_price = best_bid+level
                    if diff > 0:
                        order_list.append(LimitOrder(self.agent_id, 'ask', limit_price, diff, time))
                        l += diff
                    elif diff < 0:
                        # need to insert cancellations first in the list, such that the budget is free                         
                        order_list.insert(0, CancellationByPriceVolume(agent_id=self.agent_id, side='ask', price=limit_price, volume=-diff, time=time))
                        cancelled_volume += -diff
                    else:
                        pass

            assert cancelled_volume >= 0
            assert l >= 0
            assert m >= 0
            free_budget = cancelled_volume + (self.volume - self.active_volume) 
            assert 0 <= free_budget <= self.volume
            assert 0 <= l + m <= free_budget, 'l + m must be less than or equal to the free budget'
            assert self.volume >= self.active_volume >= cancelled_volume >= 0
            return order_list
        elif time == self.terminal_time:
            return self.sell_remaining_position(lob, time)
        else:
            return None 
    
    def get_observation(self, time, lob):        
        """
            - sets reference prices if time == self.start_time
            - computes observations 
            - returns observations

        """
        if time == self.start_time:
            self.reference_bid_price = lob.get_best_price('bid') 
            self.reference_ask_price = lob.get_best_price('ask')
            self.reference_mid_price = (self.reference_bid_price + self.reference_ask_price)/2
            
        # bid and mid price, spread
        best_bid = lob.data.best_bid_prices[-1]
        best_ask = lob.data.best_ask_prices[-1]
        mid_price = (best_bid + best_ask)/2
        bid_price_drift = (best_bid - self.reference_bid_price)/10
        mid_price_drift = (mid_price - self.reference_mid_price)/10
        spread = (best_ask - best_bid)/10        

        # volumes and imbalance 
        if self.start_at_best_price:
            bid_volumes = lob.data.bid_volumes[-1][:self.observation_book_levels]
            ask_volumes = lob.data.ask_volumes[-1][:self.observation_book_levels]        
        else:
            bid_volumes = lob.data.bid_volumes[-1][int(spread-1):self.observation_book_levels+int(spread-1)]
            ask_volumes = lob.data.ask_volumes[-1][int(spread-1):self.observation_book_levels+int(spread-1)]
        if np.sum(bid_volumes+ask_volumes) == 0:
            # print('first 5 bid+ask volumes are zero')
            # print(f'all bid vols: {lob.data.bid_volumes[-1]}')
            # print(f'all ask vols: {lob.data.ask_volumes[-1]}')
            imbalance = 0
        else:
            imbalance = np.sum(bid_volumes - ask_volumes)/np.sum(bid_volumes + ask_volumes)
            
        # normalize volumes through initial shape files 
        if self.start_at_best_price:
            bid_volumes /= self.initial_shape[int(spread-1):self.observation_book_levels+int(spread-1)]
            ask_volumes /= self.initial_shape[int(spread-1):self.observation_book_levels+int(spread-1)]
        else:
            bid_volumes /= self.initial_shape[:self.observation_book_levels]
            ask_volumes /= self.initial_shape[:self.observation_book_levels]

        # order distribution 
        volume_per_level, _ = self.get_order_allocation(lob, self.observation_book_levels, start_at_best_price=self.start_at_best_price)
        volume_per_level.append(self.volume-self.active_volume)
        order_dist = np.array(volume_per_level)
        assert np.sum(order_dist) == self.volume        
        if self.volume == 0:
            order_dist = np.array(order_dist, dtype=np.float32)
        else:
            order_dist = np.array(order_dist, dtype=np.float32)/self.volume
        
        # old one hot queue position encoding 
        # record queue position for the first two levels  
        # max_length = 50
        # max_level = 3 ## maybe change this later to self.observation_book_levels 
        # queue_positions = np.zeros((max_length, max_level), dtype=np.float32)
        # for l in range(max_level):
        #     pos = 0 
        #     price = best_bid+l+1
        #     if price in lob.price_map['ask']:
        #         for id in lob.price_map['ask'][price]:
        #             volume = lob.order_map[id].volume
        #             agent_id = lob.order_map[id].agent_id
        #             if agent_id == self.agent_id:
        #                 queue_positions[int(pos):int(pos+volume), l] = 1
        #             pos += volume        
        #     else:  
        #         pass        
        # queue_positions = queue_positions.flatten()

        # new queue position encoding 
        # TODO: volume encoding with level, queue position, and volume 
        levels = []
        queues = []
        volume_within_range = 0
        for level in range(1, self.observation_book_levels+1):
            pos = 0 
            if self.start_at_best_price:
                price = best_ask+level-1
            else:
                price = best_bid+level
            if price in lob.price_map['ask']:
                for id in lob.price_map['ask'][price]:
                    volume = lob.order_map[id].volume
                    agent_id = lob.order_map[id].agent_id
                    if agent_id == self.agent_id:
                        levels.extend(int(volume)*[level])
                        queues.extend(list(range(int(pos), int(pos+volume))))
                        volume_within_range += volume
                    pos += volume        
            else:  
                pass        
        
        max_queue_size = 40 
        # volume outside the price range or inactive 
        volume_outside = self.volume - volume_within_range
        assert 0 <= volume_outside <= self.volume
        levels.extend([self.observation_book_levels+1]*int(volume_outside))
        queues.extend([max_queue_size]*int(volume_outside))

        # filled volume 
        filled_volume = self.initial_volume - self.volume
        assert filled_volume+volume_within_range+volume_outside == self.initial_volume
        levels.extend([-self.observation_book_levels]*int(filled_volume))
        queues.extend([-max_queue_size]*int(filled_volume))

        # queues and levels 
        assert len(queues) == self.initial_volume
        assert len(levels) == self.initial_volume

        # normalization 
        queues = np.array(queues, dtype=np.float32)/max_queue_size
        levels = np.array(levels, dtype=np.float32)/(self.observation_book_levels+1)
        
        # market order imbalance 
        buys = [x for x, t in zip(lob.data.market_buy, lob.data.time_stamps) if t >= time-self.time_delta and t <= time]  
        sells = [x for x, t in zip(lob.data.market_sell, lob.data.time_stamps) if t >= time-self.time_delta and t <= time]  
        if np.sum(buys) + np.sum(sells) == 0:
            market_order_imbalance = 0
        else:
            market_order_imbalance = (np.sum(buys) - np.sum(sells))/(np.sum(buys) + np.sum(sells))
        market_order_imbalance = np.array([market_order_imbalance], dtype=np.float32)
        # limit order imbalance 
        limit_buys = [x for x, t in zip(lob.data.limit_buy, lob.data.time_stamps) if t >= time-self.time_delta and t <= time]
        limit_sells = [x for x, t in zip(lob.data.limit_sell, lob.data.time_stamps) if t >= time-self.time_delta and t <= time]
        if np.sum(limit_buys) + np.sum(limit_sells) == 0:
            limit_order_imbalance = 0
        else:
            limit_order_imbalance = (np.sum(limit_buys) - np.sum(limit_sells))/(np.sum(limit_buys) + np.sum(limit_sells))
        limit_order_imbalance = np.array([limit_order_imbalance], dtype=np.float32)
        # mid price drift from t-delta t to t 
        assert time >= lob.data.time_stamps[-1]
        idx_old = bisect.bisect_left(lob.data.time_stamps, time-self.time_delta)
        mid_price_old = (lob.data.best_bid_prices[idx_old] + lob.data.best_ask_prices[idx_old])/2 
        mid_price_now = (lob.data.best_bid_prices[-1] + lob.data.best_ask_prices[-1])/2
        deltat_drift = np.array([(mid_price_now - mid_price_old)/10], dtype=np.float32)
                        
        # concat everything into one observation 
        # 
        observation = np.array([time/self.terminal_time, self.volume/self.initial_volume, bid_price_drift, mid_price_drift, spread, imbalance], dtype=np.float32)
        observation = np.concatenate([observation, market_order_imbalance, limit_order_imbalance, deltat_drift], dtype=np.float32)
        observation = np.concatenate([observation, order_dist], dtype=np.float32)
        observation = np.concatenate([observation, bid_volumes, ask_volumes], dtype=np.float32)
        observation = np.concatenate([observation, levels, queues], dtype=np.float32)

        # final shape is 6+4+6+5+5+60+60 with observable levels = 5 
    
        return observation
    
    def get_order_allocation(self, lob, n_levels, start_at_best_price=False):
        """
            returns
                - list of volumes: [v_1, v_2,... , v_n_levels, v_>n_levels], corresponfing to best_bid+1, best_bid+2, ..., best_bid+n_levels 
                - set of order ids whose price is within [best_bid+1,..., best_bid+n_levels]

            input
                - lob: current state of the order book
                - number of levels, e.g n_levels = 3
                
            TODO: this could be a function in the order book class
            TODO: carry a separate dict in LOB which tracks orders of the agent, its position, and queue position 
        """

        best_bid = lob.get_best_price('bid')
        best_ask = lob.get_best_price('ask')
        volume_per_level = []
        orders_within_range = set()
        for level in range(1, n_levels+1):
            # 1,2,3, ..., n_levels
            # e.g. if n_levels = 4, then 1,2,3,4
            orders_on_level = 0
            if start_at_best_price:
                price = best_ask+level-1
            else:
                price = best_bid+level
            if price in lob.price_map['ask']:
                for order_id in lob.price_map['ask'][price]:
                    if lob.order_map[order_id].agent_id == self.agent_id:                                                
                        orders_on_level += lob.order_map[order_id].volume
                        orders_within_range.add(order_id)                    
                volume_per_level.append(orders_on_level)
            else:
                volume_per_level.append(0)
        # append volumes which are on prices larger than bid + n_levels 
        assert sum(volume_per_level) <= self.active_volume
        volume_per_level.append(self.active_volume-sum(volume_per_level))
        assert sum(volume_per_level) <= self.volume
        assert sum(volume_per_level) == self.active_volume
        assert sum([v < 0 for v in volume_per_level]) == 0 
        return volume_per_level, orders_within_range

    def new_event(self, time, event):
        assert event == self.agent_id
        assert time >= self.start_time
        assert time <= self.terminal_time
        assert time % self.time_delta == 0
        if time <= self.terminal_time-self.time_delta:
            return (time+self.time_delta, self.priority, self.agent_id)
        else:
            return None

    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)
                
class StrategicAgent():
    """
    - just sends limit and market orders at some frequency
    - we do not keep track of the agents position 
    """
    def __init__(self, start_time, time_delta, market_volume, limit_volume, terminal_time, rng, priority=-1) -> None:
        # assert 0 <= offset < frequency, 'offset must be in {0,1, ..., frequency-1}'        
        self.start_time = start_time
        self.time_delta = time_delta
        self.market_order_volume = market_volume
        self.limit_order_volume = limit_volume
        self.agent_id = 'strategic_agent'
        self.rng = rng
        self.direction = None 
        self.priority = priority
        self.terminal_time = terminal_time   
        return None 
    
    def generate_order(self, lob, time):        
        if (time - self.start_time) % self.time_delta == 0:
            if self.direction == 'sell':                
                limit_price = lob.get_best_price('bid')+1
                order_list = []
                order_list.append(MarketOrder(self.agent_id, 'bid', self.market_order_volume, time))
                order_list.append(LimitOrder(self.agent_id, 'ask', limit_price, self.limit_order_volume, time))                        
                return order_list
            elif self.direction == 'buy':
                limit_price = lob.get_best_price('ask')-1
                order_list = []
                order_list.append(MarketOrder(self.agent_id, 'ask', self.market_order_volume, time))
                order_list.append(LimitOrder(self.agent_id, 'bid', limit_price, self.limit_order_volume, time))                        
                return order_list
            else:
                raise ValueError(f'direction must be either buy or sell, got {self.direction}')
        else:
            # in the current set up this should never happen ! 
            return None 
        
    def reset(self):
        self.direction = self.rng.choice(['buy', 'sell'])
        return None
    
    def new_event(self, time, event):
        assert self.start_time <= time 
        assert event == self.agent_id
        if time < self.terminal_time:
            return (time+self.time_delta, self.priority, self.agent_id)
        else:
            return None
    
    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)

class InitialAgent():
    def __init__(self, start_time, initial_bid, initial_ask, initial_shape, n_initial_levels, initial_shape_file=None, priority=-1):
        self.priority = priority
        self.start_time = start_time
        self.initial_bid = initial_bid
        self.initial_ask = initial_ask
        self.agent_id = 'initial_agent'
        self.n_inital_levels = n_initial_levels
        if initial_shape_file is None:
            self.initial_shape = np.array([initial_shape]*n_initial_levels) 
        else:
            shape = np.load(initial_shape_file)
            shape = np.clip(np.rint(np.mean([shape['bidv'], shape['askv']], axis=0)), 1, np.inf) 
            assert self.n_inital_levels <= len(shape), 'initial shape file does not have enough levels'
            self.initial_shape = shape[:n_initial_levels]
        return None 
    
    def generate_order(self, lob, time):
        # lob is part of the argumenets because all the other generate_order methods have lob as an argument 
        # TODO: how to avoid this ? 
        assert time == self.start_time
        orders = []
        for idx, price in enumerate(np.arange(self.initial_ask-1, self.initial_ask-1-self.n_inital_levels, -1)):
            order = LimitOrder(agent_id=self.agent_id, side='bid', price=price, volume=self.initial_shape[idx], time=time)
            orders.append(order)
        for idx, price in enumerate(np.arange(self.initial_bid+1, self.initial_bid+1+self.n_inital_levels, 1)):
            order = LimitOrder(agent_id=self.agent_id, side='ask', price=price, volume=self.initial_shape[idx], time=time)
            orders.append(order)
        return orders
    
    def new_event(self, time, event):
        # time is passed here because all other methods have time 
        assert time == self.start_time
        assert event == self.agent_id
        return None 
    
    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)

    def reset(self):
        return None

class TestAgent():
    def __init__(self, start_time=-1, terminal_time=4, time_delta=1, priority=1, fills=True):
        self.start_time = start_time
        self.terminal_time = terminal_time
        self.time_delta = time_delta
        self.agent_id = 'test_agent'
        self.priority = priority
        self.fills = fills 
        return None
    
    def generate_order(self, lob, time):
        assert time >= self.start_time
        if time == self.start_time:
            orders = []
            orders.append(LimitOrder(self.agent_id, 'bid', 100, 1, time))
            orders.append(LimitOrder(self.agent_id, 'ask', 101, 1, time))
            return orders
        elif time <= self.terminal_time:
            best_bid = lob.get_best_price('bid')
            orders = []
            if self.fills:
                orders.append(MarketOrder(self.agent_id, 'ask', 2, time))
                orders.append(LimitOrder(self.agent_id, 'bid', best_bid+1, 1, time))
                orders.append(LimitOrder(self.agent_id, 'ask', best_bid+2, 1, time))
            else:
                orders.append(LimitOrder(self.agent_id, 'bid', best_bid, 1, time))
            return orders
        else:
            raise ValueError('should not happen')
    
    def new_event(self, time, event):
        assert event == self.agent_id
        assert time >= self.start_time
        assert time <= self.terminal_time
        if time < self.terminal_time:
            return (time+self.time_delta, self.priority, self.agent_id)
        else:
            return None
    
    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)
    
    def reset(self):
        pass 

class ObservationAgent():
    def __init__(self, start_time, time_delta, terminal_time, priority, agent_id):
        self.start_time = start_time
        self.time_delta = time_delta
        self.terminal_time = terminal_time
        self.priority = priority
        self.agent_id = agent_id
    
    def reset(self):
        pass
    
    def generate_order(self, lob, time):
        return None
    
    def initial_event(self):
        return (self.start_time, self.priority, self.agent_id)
    
    def new_event(self, time, event):
        assert time >= self.start_time
        assert time <= self.terminal_time
        assert event == self.agent_id
        if time < self.terminal_time:
            return (time+self.time_delta, self.priority, self.agent_id)
        else:
            return None