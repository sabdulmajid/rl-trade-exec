from copy import deepcopy
from operator import neg
from sortedcontainers import SortedDict, SortedList
import numpy as np
import pandas as pd
import warnings

class DynamicDict:
    def __init__(self):
        self.messages = {}
    def add(self, agent_id, message):
        if agent_id in self.messages:
            self.messages[agent_id].append(message)
        else:
            self.messages[agent_id] = [message]

# order types
class Order:
    def __init__(self, agent_id, type, time):
        self.agent_id = agent_id
        self.type = type
        self.time = time

class LimitOrder(Order):
    def __init__(self, agent_id, side, price, volume, time):
        super().__init__(agent_id, 'limit', time)
        if volume is None:
            raise ValueError(f"volume is None for order at time={time}, sides={side}, agent={agent_id}")
        assert side in ['bid', 'ask'], "side must be bid or ask"
        assert volume > 0, f"volume must be positive, side={side}, volume={volume}, agent_id={agent_id}, price={price}, time={time}"
        self.side = side
        self.price = price
        self.volume = volume
        self.order_id = None
        self.type = 'limit'
    def __repr__(self):
        return f'LO(agent: {self.agent_id}, side: {self.side}, price: {self.price}, volume: {self.volume}, order_id: {self.order_id}, time: {self.time})'

class MarketOrder(Order):
    def __init__(self, agent_id, side, volume, time):
        super().__init__(agent_id, 'market', time)
        assert side in ['bid', 'ask'], "side must be bid or ask"
        assert volume > 0, "volume must be positive"
        self.side = side
        self.volume = volume
        self.type = 'market'
    def __repr__(self):
        return f'MO(side: {self.side}, volume: {self.volume}, time: {self.time})'

class CancellationByPriceVolume(Order):
    def __init__(self, agent_id, side, price, volume, time):
        super().__init__(agent_id, 'cancellation_by_price_volume', time)
        assert side in ['bid', 'ask'], "side must be bid or ask"
        assert volume > 0, "volume must be positive"
        self.side = side
        self.price = price
        self.volume = volume
        self.type = 'cancellation_by_price_volume'
    def __repr__(self):
        return f'CBPV(agent: {self.agent_id}, side: {self.side}, price: {self.price}, volume: {self.volume}, time: {self.time})'

class Cancellation(Order):
    def __init__(self, agent_id, order_id, time):
        super().__init__(agent_id, 'cancellation', time)
        assert order_id >= 0, "order id must be positive"
        self.order_id = order_id
        self.agent_id = agent_id
        self.type = 'cancellation'
    def __repr__(self):
        return f'Cancellation(agent_id={self.agent_id}, order_id={self.order_id})'

class Modification(Order):
    def __init__(self, agent_id, order_id, new_volume, time):
        super().__init__(agent_id, 'modification', time)
        assert order_id >= 0, "order id must be positive"
        assert new_volume > 0, "volume must be positive"
        self.order_id = order_id
        self.volume = new_volume
        self.type = 'modification'
    def __repr__(self):
        return f'Modification(agent_id={self.agent_id}, order_id={self.order_id}, volume={self.new_volume}, time={self.time})'

# confirmation messages
class ModificationConfirmation():
    def __init__(self, order, new_volume, old_volume):
        assert new_volume <= old_volume, "new volume must be smaller than old volume"
        self.order = order
        self.new_volume = new_volume
        self.old_volume = old_volume
        self.type = 'modification'
    def __repr__(self) -> str:
        return f'ModificationConf(agent_id={self.order.agent_id}, order_id={self.order.order_id}, new_volume={self.new_volume}, old_volume={self.old_volume}, time={self.order.time})'


class LimitOrderFill():
    def __init__(self, order_id, price, volume, side, agent_id):
        self.order_id = order_id
        self.price = price
        self.volume = volume
        self.side = side
        self.agent_id = agent_id
        self.type = 'limit'
    def __repr__(self) -> str:
        return f'FillLO(order_id={self.order_id}, price={self.price}, volume={self.volume}, side={self.side}, agent_id={self.agent_id})'
    

class PassiveFill():
    def __init__(self, order, filled_volume, partial_fill):
        assert filled_volume > 0, "filled volume must be positive"
        self.order = order 
        self.filled_volume = filled_volume
        self.partial_fill = partial_fill
        self.type = 'passive_limit'
    def __repr__(self) -> str:
        tag = 'partial_fill' if self.partial_fill else 'full_fill'
        return f'PassiveFill(filled_volume={self.filled_volume}, price={self.order.price}, id={self.order.order_id}, side={self.order.side}, {tag})'

class MarketOrderFill():
    def __init__(self, order, filled_volume, execution_price, partial_fill, passive_fills):
        self.order = order
        self.filled_volume = filled_volume
        self.execution_price = execution_price
        self.passive_fills = passive_fills
        self.partial_fill = partial_fill
        self.type = 'market'
    def __repr__(self) -> str:
        return f'FillMO(filled_volume={self.filled_volume}, execution_price={self.execution_price}, side={self.order.side})'


class CancellationMessage():
    def __init__(self, order, price, side, volume):
        self.order = order
        self.price = price
        self.side = side
        self.volume = volume
        self.type = 'cancellation'
    def __repr__(self) -> str:
        return f'CancellationConf(order_id={self.order.order_id}, agent_id={self.order.agent_id}, side={self.side}, price={self.price}, volume={self.volume})'

class CancellationByPriceVolumeMessage():
    def __init__(self, order, affected_orders, filled_volume, partial_fill, price):
        self.order = order 
        self.affected_orders = affected_orders
        self.filled_volume = filled_volume
        self.partial_fill = partial_fill        
        self.price = price
        self.type = 'cancellation_by_price_volume'
    def __repr__(self) -> str:
        tag = 'partial_fill' if self.partial_fill else 'full_fill'
        return f'CancellationPV(filled_volume={self.filled_volume}, price={self.price}, {tag})'


class Data():
    def __init__(self, level) -> None:
        self.level = level 
        self.reset()
    
    def reset(self):
        self.orders = []
        # bid/ask prices and volumes up to a certain level 
        self.bid_prices = []
        self.ask_prices = []
        self.bid_volumes = []
        self.ask_volumes = []
        # best bid/ask prices and volumes
        self.best_bid_prices = []
        self.best_ask_prices = []
        self.best_bid_volumes = []
        self.best_ask_volumes = []
        # market trades 
        self.market_buy = []
        self.market_sell = []
        # 
        self.limit_buy = []
        self.limit_sell = []
        # 
        self.time_stamps = []
    

class LimitOrderBook:
    def __init__(self, list_of_agents = [], level=10, only_volumes=False):
        # how many levels of the order book are stored. level=10 means that the first 10 levels of the order book are logged? 
        self.level = level
        self.registered_agents = list_of_agents
        # order ids by agent 
        if not only_volumes:
            self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
            self.order_map = {}
            self.order_map_by_agent = {agent_id: set() for agent_id in list_of_agents}
        self.update_n = 0
        # order matters here !!!, first data, then logging 
        self.data = Data(level=level)
        self.price_volume_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
        self.only_volumes = only_volumes
        self.log_everything = True
        self.time = -np.inf 
        # self.only_shape = only_shape    
        # TODO: only shape options. where we only keep track of the shape. 
        # initialize state of the order book at step n = 0  
        # self._logging()
    
    def _logging(self, order=None):
        # ToDo: increase efficiency of logging
        self.data.orders.append(order)
        # level 2 data including empty levels
        bid_prices, bid_volumes = self.level2('bid')
        ask_prices, ask_volumes = self.level2('ask')
        best_bid = self.get_best_price('bid')
        best_ask = self.get_best_price('ask')
        best_bid_volume = self.volume_at_price('bid', best_bid) 
        best_ask_volume = self.volume_at_price('ask', best_ask)
        if order.time > self.time:
            self.data.time_stamps.append(order.time)
            # set up new elements in lists 
            self.data.bid_prices.append(bid_prices)
            self.data.ask_prices.append(ask_prices)
            self.data.bid_volumes.append(bid_volumes)
            self.data.ask_volumes.append(ask_volumes)
            # best bid/ask prices and volumes for easy look up 
            self.data.best_bid_prices.append(best_bid)
            self.data.best_ask_prices.append(best_ask)
            self.data.best_bid_volumes.append(best_bid_volume)
            self.data.best_ask_volumes.append(best_ask_volume)
            # log market order sizes 
            if order.type == 'market':
                if order.side == 'bid':
                    self.data.market_sell.append(order.volume)
                    self.data.market_buy.append(0)
                else:
                    self.data.market_buy.append(order.volume)
                    self.data.market_sell.append(0)
            else:
                self.data.market_buy.append(0)
                self.data.market_sell.append(0)
            if order.type == 'limit':
                if order.side == 'bid':
                    self.data.limit_buy.append(order.volume)
                    self.data.limit_sell.append(0)
                else:
                    self.data.limit_sell.append(order.volume)
                    self.data.limit_buy.append(0)
            else:
                self.data.limit_buy.append(0)
                self.data.limit_sell.append(0)
                
        else: 
            # update last element in lists
            self.data.bid_prices[-1] = bid_prices
            self.data.ask_prices[-1] = ask_prices
            self.data.bid_volumes[-1] = bid_volumes
            self.data.ask_volumes[-1] = ask_volumes
            self.data.best_bid_prices[-1] = best_bid
            self.data.best_ask_prices[-1] = best_ask
            self.data.best_bid_volumes[-1] = best_bid_volume
            self.data.best_ask_volumes[-1] = best_ask_volume
            # log market order sizes 
            if order.type == 'market':
                if order.side == 'bid':
                    self.data.market_sell[-1] += order.volume
                else:
                    self.data.market_buy[-1] += order.volume
            if order.type == 'limit':
                if order.side == 'bid':
                    self.data.limit_buy[-1] += order.volume
                else:
                    self.data.limit_sell[-1] += order.volume
        
        self.time = order.time

        # if np.isnan(best_bid):
        #     self.data.best_bid_volumes.append(np.nan)
        # else:
        # if np.isnan(best_ask):
        #     self.data.best_ask_volumes.append(np.nan)
        # else:
        #     self.data.best_ask_volumes.append(self.price_volume_map['ask'][best_ask])
    

    def process_order(self, order):
        """
        - an order is a dictionary with fields agent_id, type, side, price, volume, order_id
        - some of those fields are optional depending on the order type 
        """

        # agent_id should be one of the registered agents 
        assert order.agent_id in self.registered_agents, "agent id not registered"
        assert order.time >= self.time, "time must be greater than current time"

        if order.type == 'limit':
            msg = self.handle_limit_order(order)
        elif order.type == 'market':
            msg = self.handle_market_order(order)
        elif order.type == 'cancellation':
            msg = self.cancellation(order)
        elif order.type == 'modification':
            msg = self.modification(order)
        elif order.type == 'cancellation_by_price_volume':
            msg = self.cancellation_by_price_volume(order)
        else:
            raise ValueError("order type not supported")

        self.update_n += 1 
        # log shape of the book after transition 
        if self.log_everything:        
            self._logging(order)
        else:
            pass
        return msg


    def handle_limit_order(self, order):      
        """
        - if limit price is in price map, add volume to the price level
        - else create a new price level with the volume

        Args:
            order is dict with keys (type, side, price, volume)
        
        Returns:
            None. Changes the state of the order book internally
                - "order_id" is assigned to the order
                - limit order is added to the order map under the key "order_id" and with the whole dict order as value 
                - limit order is added to the price map under the key "price" and with order_id as value
        """

        # only do this check if the opposite side is not empty
        if order.side == 'ask' and self.price_volume_map['bid']:
            assert order.price > self.get_best_price('bid'), "sent ask limit order with price <= bid price"
        if order.side == 'bid' and self.price_volume_map['ask']:    
            assert order.price < self.get_best_price('ask'), "sent bid limit order with price >= ask price"
        
        # add order to price volume map, return if we only keep track of volumes
        if order.price in self.price_volume_map[order.side]:
            self.price_volume_map[order.side][order.price] += order.volume
        else:
            self.price_volume_map[order.side][order.price] = order.volume        
        if self.only_volumes:
            return None 
        
        if order.order_id:
            raise ValueError("limit order should have no order id")
        else:
            order.order_id = self.update_n            

        
        if order.price in self.price_map[order.side]:
            # add order to price level 
            self.price_map[order.side][order.price].add(order.order_id) 
        else:
            # SortedList 
            self.price_map[order.side][order.price] = SortedList([order.order_id])
        
        # add order to order map and order map by agent
        self.order_map[order.order_id] = order
        self.order_map_by_agent[order.agent_id].add(order.order_id)

        return LimitOrderFill(order_id=order.order_id, price=order.price, volume=order.volume, side=order.side, agent_id=order.agent_id)


    def handle_market_order(self, order):
        """
        - match order against limit order in the book
        - return profit message to both agents 
        - modify the state of filled orders in the book 
        """
        assert order.side in ['bid', 'ask'], "side must be either bid or ask"
        assert order.volume > 0, "volume must be positive"
        assert order.agent_id in self.registered_agents, "agent id not registered"        

        side = order.side 

        if not self.price_volume_map[side]:
            raise ValueError(f"{side} side is empty!")
        
        # update price volume map
        remaining_market_volume = order.volume
        prices = list(self.price_volume_map[side].keys())
        for price in prices: 
            diff = self.price_volume_map[side][price]-remaining_market_volume
            self.price_volume_map[side][price] = max(diff, 0)
            remaining_market_volume = max(-diff, 0)
            if self.price_volume_map[side][price] == 0:
                self.price_volume_map[side].pop(price)
            if remaining_market_volume == 0.0:                
                assert diff >= 0
                break
        if remaining_market_volume > 0:
            warnings.warn("market volume not fully executed\n"
                          f"order time: {order.time}\n"
                          f"order volume: {order.volume}\n"
                          f"bid volumes: {self.level2('bid')[1][:10]}\n"
                          f"ask volumes: {self.level2('ask')[1][:10]}")
        if self.only_volumes:
            return None


        market_volume = order.volume 
        execution_price = 0.0
        filled_volume = 0.0 
        # list of fill messages for each agent         
        passive_fills = DynamicDict()

        # remaining_market_volume = order.volume
        # prices = list(self.price_map[side].keys())
        for price in prices: 
            # cp = counterparty             
            cp_order_ids = deepcopy(self.price_map[side][price])
            for cp_order_id in cp_order_ids:
                cp_order = self.order_map[cp_order_id]
                cp_agent_id = cp_order.agent_id
                if market_volume < 0:
                    raise ValueError("market volume is negative")
                elif market_volume < cp_order.volume:
                    # counterparty order is partially filled
                    # the partial fill tag might not be necessary. mainly just for additional info. 
                    msg = PassiveFill(order=cp_order, filled_volume=market_volume, partial_fill=True)
                    passive_fills.add(cp_agent_id, msg)
                    cp_order.volume -= market_volume
                    execution_price += price * market_volume
                    filled_volume += market_volume
                    market_volume = 0.0
                    break
                elif market_volume >= cp_order.volume:
                    # counterparty order is fully filled
                    # fill_msg = PassiveFill(ord, filled_volume=cp_order.volume, fill_price=price, side=side, partial_fill=False, agent_id=cp_agent_id)
                    msg = PassiveFill(order=cp_order, filled_volume=cp_order.volume, partial_fill=False)
                    passive_fills.add(cp_agent_id, msg)
                    self.price_map[side][price].remove(cp_order_id)              
                    self.order_map.pop(cp_order_id)
                    self.order_map_by_agent[cp_order.agent_id].remove(cp_order.order_id)   # remove is for sets 
                    filled_volume += cp_order.volume
                    execution_price += price * cp_order.volume
                    market_volume = market_volume - cp_order.volume
                    if market_volume == 0.0:
                        break
                else:
                    raise ValueError("this should not happen")
            # if no more orders are left on the level, remove the entire price level
            if not self.price_map[side][price]:
                self.price_map[side].pop(price)
                assert price not in self.price_volume_map[side], "price still in price volume map"
            if market_volume == 0.0:                
                break
        
        # basic checks 
        # warnings.warn(f"{order}, market volume not fully executed")
        if market_volume > 0.0: 
            warnings.warn(f"{order}, market volume not fully executed")
            # print(f'order time: {order.time}')
            # print(f'bid volumes: {self.level2("bid")[1][:10]}')
            # print(f'ask volumes: {self.level2("ask")[1][:10]}')
        if market_volume < 0:
            raise ValueError("filled volume cannot be negative")
        
        # create fill message for the market order
        msg = MarketOrderFill(order = order, execution_price=execution_price, filled_volume=filled_volume, partial_fill=market_volume>0, passive_fills=passive_fills.messages)

        return msg


    def cancellation(self, order):
        assert not self.only_volumes, "only volumes option with cancellation not supported"
        assert order.agent_id in self.order_map_by_agent, "order id not in order map by agent"
        assert order.order_id in self.order_map, "order id not in order map"
        assert order.agent_id == self.order_map[order.order_id].agent_id, "agent id does not match order id"
        # select id, side, price 
        id = order.order_id
        # for info 
        price = self.order_map[order.order_id].price
        volume = self.order_map[order.order_id].volume
        side = self.order_map[order.order_id].side
        # update price volume map
        self.price_volume_map[side][price] -= volume
        assert self.price_volume_map[side][price] >= 0, "price volume map is negative"
        if self.price_volume_map[side][price] == 0:
            self.price_volume_map[side].pop(price)
        # remove 
        self.price_map[side][price].remove(id)
        self.order_map.pop(id)
        self.order_map_by_agent[order.agent_id].remove(id)       
        # delete price level if empty
        if not self.price_map[side][price]:
            self.price_map[side].pop(price)
        return CancellationMessage(order=order, price=price, side=side, volume=volume)
    
    def cancellation_by_price_volume(self, order):
        assert order.agent_id in self.registered_agents, "agent id not registered"
        assert order.side in ['bid', 'ask'], "side must be either bid or ask"
        assert order.price in self.price_volume_map[order.side], "price not in price map"
        assert order.volume >= 0, "volume must be positive"

        if self.only_volumes: 
            # update price volume map here, otherwise price volume map is updated in the loop below 
            self.price_volume_map[order.side][order.price]  = max(self.price_volume_map[order.side][order.price]-order.volume, 0)
            if self.price_volume_map[order.side][order.price] == 0:
                self.price_volume_map[order.side].pop(order.price)
            return None

        volume = order.volume
        affected_orders = []
        order_list = []

        # go through the order ids in reverse order
        # ToDo: change this to update the book directly 
        for cp_order_id in self.price_map[order.side][order.price][::-1]:
            if volume == 0:
                break
            cp_order = self.order_map[cp_order_id]
            if cp_order.agent_id == order.agent_id:
                if volume < 0:
                    raise ValueError("cancellation volume is negative")
                elif volume < cp_order.volume:
                    order_list.append(Modification(agent_id=order.agent_id, order_id=cp_order_id, new_volume=cp_order.volume-volume, time=order.time)) 
                    volume = 0.0
                    break
                elif volume >= cp_order.volume:
                    cancellation = Cancellation(order.agent_id, cp_order_id, time=order.time)                    
                    order_list.append(cancellation)
                    volume = volume - cp_order.volume
            else:
                pass 
        
        msg_list = [self.process_order(order) for order in order_list]

        return CancellationByPriceVolumeMessage(order=order, affected_orders=msg_list, filled_volume=order.volume-volume, price=order.price, partial_fill=volume>0)

    def process_order_list(self, order_list):
        """
        - process a list of orders 
        - return a list of messages 
        """
        return [self.process_order(order) for order in order_list]


    def modification(self, order):
        assert not self.only_volumes, "only volumes option with modification not supported"
        assert order.volume <= self.order_map[order.order_id].volume, "new volume larger than original order volume"
        assert 0 < order.volume, "volume must be positive"
        # update volume 
        old_volume = self.order_map[order.order_id].volume
        self.order_map[order.order_id].volume = order.volume
        # update price volume map 
        diff = old_volume - order.volume
        assert diff > 0, "diff must be positive"
        self.price_volume_map[self.order_map[order.order_id].side][self.order_map[order.order_id].price] -= diff
        return ModificationConfirmation(order=order, new_volume=order.volume, old_volume=old_volume)
    
    def get_best_price(self, side):
        if not self.price_volume_map[side]:
            return np.nan
        else: 
            return self.price_volume_map[side].keys()[0]
    
    def level2(self, side):
        """
        why not lists ? 
        if side == 'bid':
            if side is empty:
                - best bid prices = np.empty(level)
                - best bid volumes = np.empty(level)
            else:
                - best bid prices up to level: [p_1, p_2, ... ,p_level]
                - np array of best bid volumes up to level: [v_1, v_2, ... ,v_level]
                - includes empty price levels
            return (bid_prices, bid_volumes)
        """        
        assert side in ['bid', 'ask'], "side must be either bid or ask"

        if side == 'ask':
            opposite_side = 'bid'  
            sign = 1
        else:
            opposite_side = 'ask'
            sign = -1
        

        # opposite side is empty
        if not self.price_volume_map[opposite_side]:
            # side empty
            if not self.price_volume_map[side]:
                return np.empty(self.level)*np.nan, np.empty(self.level)*np.nan
            # side not empty 
            else:                
                # start counting from best price
                best_price = self.get_best_price(side)
                prices = np.arange(best_price, best_price+sign*self.level, sign)
                # volumes = np.array([self.volume_at_price(side, price) for price in prices])
                volumes = [self.price_volume_map[side][price] if price in self.price_volume_map[side] else 0 for price in prices]
                volumes = np.array(volumes)
                return prices, volumes
        # opposite side not empty
        else:
            best_price = self.get_best_price(opposite_side)
            prices = np.arange(best_price+sign, best_price + sign + sign*self.level, sign)
            volumes = [self.price_volume_map[side][price] if price in self.price_volume_map[side] else 0 for price in prices]
            volumes = np.array(volumes)
            return prices, volumes

    
    def volume_at_price(self, side, price):
        if price not in self.price_volume_map[side]:
            return np.nan 
        else:
            return self.price_volume_map[side][price] 
    
    def find_queue_position(self, order_id):        
        # note, it is not entirely clear, what a queue position of an order of lot size > 1 should be
        # we just take the first occurence as its queue position 
        # implement all levels option 
        if order_id not in self.order_map:
            raise ValueError('order_id not found on this side of the book')
        order = self.order_map[order_id]        
        queue_position = 0
        level = self.price_map[order.side][order.price]
        for id in level:
            if id == order_id:
                return queue_position
            queue_position += self.order_map[id].volume
        raise ValueError('order_id not found on this price level')
    
    def clear_orders(self, level):
        """
        - cancel all orders beyond a certain level 
        - don't send confirmation messages
        """
        best_bid = self.get_best_price('bid')   
        best_ask = self.get_best_price('ask')

        # update prices volume map 
        for price in self.price_volume_map['bid']:
            if price < best_bid - level:
                self.price_volume_map['bid'].pop(price)
        for price in self.price_volume_map['ask']:
            if price > best_ask + level:
                self.price_volume_map['ask'].pop(price)

        if self.only_volumes:
            return None

        for price in self.price_map['bid']:
            if price < best_bid - level:
                for order_id in self.price_map['bid'][price]:
                    self.order_map_by_agent[self.order_map[order_id].agent_id].remove(order_id)
                    self.order_map.pop(order_id)
                self.price_map['bid'].pop(price)
        for price in self.price_map['ask']:
            if price > best_ask + level:
                for order_id in self.price_map['ask'][price]:
                    self.order_map_by_agent[self.order_map[order_id].agent_id].remove(order_id)
                    self.order_map.pop(order_id)
                self.price_map['ask'].pop(price)
        
        return


    def log_to_df(self):
        data = {'best_bid_price': self.data.best_bid_prices, 'best_ask_price': self.data.best_ask_prices, 'best_bid_volume': self.data.best_bid_volumes, 'best_ask_volume': self.data.best_ask_volumes}
        # level 2 data
        bid_prices = np.vstack(self.data.bid_prices)
        ask_prices = np.vstack(self.data.ask_prices)
        bid_volumes = np.vstack(self.data.bid_volumes)
        ask_volumes = np.vstack(self.data.ask_volumes)
        for i in range(0, self.level):
            data[f'bid_price_{i}'] = bid_prices[:,i]
            data[f'bid_volume_{i}'] = bid_volumes[:,i]
            data[f'ask_price_{i}'] = ask_prices[:,i]
            data[f'ask_volume_{i}'] = ask_volumes[:,i]
        data['time'] = self.data.time_stamps
        data = pd.DataFrame.from_dict(data)
        # orders
        orders = {}
        order_type = ['M' if x.type == 'market' else 'L' if x.type == 'limit' else 'C' if x.type == 'cancellation' else 'PC' if x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]
        order_side = [x.side if x.type == 'limit' or x.type == 'market' or x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]
        order_size = [x.volume if x.type == 'limit' or x.type == 'market' or x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]
        order_price = [x.price if x.type == 'limit' or x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]        
        order_time = [x.time for x in self.data.orders]
        orders['type'] = order_type
        orders['side'] = order_side
        orders['size'] = order_size
        orders['price'] = order_price        
        orders['time'] = order_time
        orders = pd.DataFrame(orders)
        # 
        market_orders = {}
        market_orders['buy'] = self.data.market_buy
        market_orders['sell'] = self.data.market_sell
        market_orders['time'] = self.data.time_stamps
        market_orders = pd.DataFrame(market_orders)
        return data, orders, market_orders
        


if __name__ == "__main__":
    LOB = LimitOrderBook(smart_agent_id='smart_agent', noise_agent_id='noise_agent')
    lo = LimitOrder('noise_agent', 'bid', 100, 10)
    msg = LOB.process_order(lo)
    print(msg)
    lo = LimitOrder('noise_agent', 'ask', 101, 10)
    msg = LOB.process_order(lo)
    print(msg)
    lo = LimitOrder('noise_agent', 'bid', 99, 10)
    msg = LOB.process_order(lo)
    print(msg)
    p = LOB.find_queue_position(lo.order_id)
    mo = MarketOrder('smart_agent', 'bid', 12)
    msg = LOB.process_order(mo)
    # assert msg['execution_price'] == msg['filled_orders']['noise_agent'][0]['fill_price']*msg['filled_orders']['noise_agent'][0]['filled_volume'] + msg['filled_orders']['noise_agent'][1]['fill_price']*msg['filled_orders']['noise_agent'][1]['filled_volume']    
    print(msg)
    lo = LimitOrder('noise_agent', 'bid', 94, 10)
    msg = LOB.process_order(lo)
    lo = LimitOrder('noise_agent', 'bid', 95, 3)
    msg = LOB.process_order(lo)
    print(LOB.order_map)
    c = Cancellation('noise_agent', 5)
    msg = LOB.process_order(c)
    m = Modification('noise_agent', 4, 5)
    msg = LOB.process_order(m)
    m = Modification('noise_agent', 4, 2)
    msg = LOB.process_order(m)
    out = LOB.level2('bid', level=5)
    print(LOB.price_map)
    print('done')
