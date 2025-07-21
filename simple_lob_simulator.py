import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class Order:
    """Simple order representation"""
    order_id: int
    agent_id: str
    side: str  # 'bid' or 'ask'
    price: float
    volume: int
    time: float
    order_type: str = 'limit'  # 'limit' or 'market'

class SimpleLOB:
    """Simplified Limit Order Book for demonstration"""
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids = defaultdict(list)  # price -> list of orders
        self.asks = defaultdict(list)  # price -> list of orders
        self.order_id_counter = 0
        self.time = 0
        self.trade_history = []
        
    def add_order(self, agent_id: str, side: str, price: float, volume: int) -> int:
        """Add a limit order to the book"""
        self.order_id_counter += 1
        order = Order(
            order_id=self.order_id_counter,
            agent_id=agent_id,
            side=side,
            price=price,
            volume=volume,
            time=self.time
        )
        
        if side == 'bid':
            self.bids[price].append(order)
        else:
            self.asks[price].append(order)
            
        return self.order_id_counter
    
    def market_order(self, agent_id: str, side: str, volume: int) -> Dict:
        """Execute a market order"""
        executed_volume = 0
        total_cost = 0.0
        fills = []
        
        if side == 'buy':
            # Buy market order - consume asks (starting from best ask)
            sorted_prices = sorted(self.asks.keys())
            for price in sorted_prices:
                if executed_volume >= volume:
                    break
                    
                while self.asks[price] and executed_volume < volume:
                    order = self.asks[price][0]
                    fill_volume = min(order.volume, volume - executed_volume)
                    
                    # Record the fill
                    fills.append({
                        'price': price,
                        'volume': fill_volume,
                        'passive_agent': order.agent_id
                    })
                    
                    executed_volume += fill_volume
                    total_cost += price * fill_volume
                    
                    # Update or remove the passive order
                    if fill_volume >= order.volume:
                        self.asks[price].pop(0)
                    else:
                        order.volume -= fill_volume
                        
                    if not self.asks[price]:
                        del self.asks[price]
        
        else:  # sell
            # Sell market order - consume bids (starting from best bid)
            sorted_prices = sorted(self.bids.keys(), reverse=True)
            for price in sorted_prices:
                if executed_volume >= volume:
                    break
                    
                while self.bids[price] and executed_volume < volume:
                    order = self.bids[price][0]
                    fill_volume = min(order.volume, volume - executed_volume)
                    
                    fills.append({
                        'price': price,
                        'volume': fill_volume,
                        'passive_agent': order.agent_id
                    })
                    
                    executed_volume += fill_volume
                    total_cost += price * fill_volume
                    
                    if fill_volume >= order.volume:
                        self.bids[price].pop(0)
                    else:
                        order.volume -= fill_volume
                        
                    if not self.bids[price]:
                        del self.bids[price]
        
        avg_price = total_cost / executed_volume if executed_volume > 0 else 0
        
        # Record trade
        trade = {
            'time': self.time,
            'agent_id': agent_id,
            'side': side,
            'volume': executed_volume,
            'avg_price': avg_price,
            'fills': fills
        }
        self.trade_history.append(trade)
        
        return trade
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return best_bid, best_ask
    
    def get_market_depth(self, levels: int = 5) -> Dict:
        """Get market depth up to specified levels"""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        bid_data = []
        for price in bid_prices:
            volume = sum(order.volume for order in self.bids[price])
            bid_data.append({'price': price, 'volume': volume})
            
        ask_data = []
        for price in ask_prices:
            volume = sum(order.volume for order in self.asks[price])
            ask_data.append({'price': price, 'volume': volume})
            
        return {
            'bids': bid_data,
            'asks': ask_data,
            'best_bid': bid_prices[0] if bid_prices else None,
            'best_ask': ask_prices[0] if ask_prices else None
        }
    
    def advance_time(self, dt: float = 1.0):
        """Advance the simulation time"""
        self.time += dt

class SimpleMarketSimulator:
    """Simple market simulator with noise traders"""
    
    def __init__(self, initial_mid_price: float = 100.0, tick_size: float = 0.01):
        self.lob = SimpleLOB(tick_size)
        self.mid_price = initial_mid_price
        self.noise_agents = [f"noise_{i}" for i in range(10)]
        self.initialize_book()
        
    def initialize_book(self):
        """Initialize the order book with some liquidity"""
        np.random.seed(42)  # For reproducible demo
        
        # Add initial orders around mid price
        for i in range(1, 6):  # 5 levels on each side
            bid_price = self.mid_price - i * self.lob.tick_size
            ask_price = self.mid_price + i * self.lob.tick_size
            
            # Random volumes
            bid_vol = max(10, int(100 * np.exp(-0.3 * i) + np.random.normal(0, 10)))
            ask_vol = max(10, int(100 * np.exp(-0.3 * i) + np.random.normal(0, 10)))
            
            # Add orders from noise traders
            agent_bid = random.choice(self.noise_agents)
            agent_ask = random.choice(self.noise_agents)
            
            self.lob.add_order(agent_bid, 'bid', bid_price, bid_vol)
            self.lob.add_order(agent_ask, 'ask', ask_price, ask_vol)
    
    def step(self) -> Dict:
        """Simulate one time step"""
        # Random events from noise traders
        events = []
        
        # 30% chance of new limit order
        if np.random.random() < 0.3:
            agent = random.choice(self.noise_agents)
            side = random.choice(['bid', 'ask'])
            
            best_bid, best_ask = self.lob.get_best_bid_ask()
            
            if side == 'bid' and best_bid:
                # Place bid near current best
                price = best_bid + random.choice([-1, 0, 1]) * self.lob.tick_size
                price = max(price, self.mid_price - 5 * self.lob.tick_size)
            elif side == 'ask' and best_ask:
                # Place ask near current best
                price = best_ask + random.choice([-1, 0, 1]) * self.lob.tick_size
                price = min(price, self.mid_price + 5 * self.lob.tick_size)
            else:
                price = self.mid_price + random.choice([-1, 1]) * self.lob.tick_size
            
            volume = max(5, int(np.random.exponential(30)))
            self.lob.add_order(agent, side, price, volume)
            events.append(f"Added {side} order: {volume}@{price:.2f}")
        
        # 10% chance of market order
        if np.random.random() < 0.1:
            agent = random.choice(self.noise_agents)
            side = random.choice(['buy', 'sell'])
            volume = max(1, int(np.random.exponential(20)))
            
            trade = self.lob.market_order(agent, side, volume)
            if trade['volume'] > 0:
                events.append(f"Market {side}: {trade['volume']}@{trade['avg_price']:.2f}")
        
        # Update mid price based on recent trades
        if self.lob.trade_history:
            recent_trade = self.lob.trade_history[-1]
            # Small price impact
            impact = 0.001 * recent_trade['volume'] * (1 if recent_trade['side'] == 'buy' else -1)
            self.mid_price += impact
        
        # Add some random walk
        self.mid_price += np.random.normal(0, 0.005)
        
        self.lob.advance_time()
        
        return {
            'events': events,
            'mid_price': self.mid_price,
            'depth': self.lob.get_market_depth(),
            'time': self.lob.time
        }

class SimpleExecutionAgent:
    """Simple execution agent for demonstration"""
    
    def __init__(self, agent_id: str, total_volume: int, strategy: str = 'twap'):
        self.agent_id = agent_id
        self.total_volume = total_volume
        self.remaining_volume = total_volume
        self.strategy = strategy
        self.executed_trades = []
        self.start_time = None
        
    def get_action(self, market_state: Dict, current_time: float) -> Dict:
        """Decide what action to take"""
        if self.start_time is None:
            self.start_time = current_time
            
        if self.remaining_volume <= 0:
            return {'type': 'none'}
        
        depth = market_state['depth']
        
        if self.strategy == 'market':
            # Execute everything as market order
            return {
                'type': 'market',
                'side': 'sell',
                'volume': self.remaining_volume
            }
            
        elif self.strategy == 'twap':
            # Time-weighted average price - execute in chunks
            time_elapsed = current_time - self.start_time
            execution_rate = 10  # lots per time unit
            target_executed = min(self.total_volume, execution_rate * time_elapsed)
            volume_to_execute = max(0, target_executed - (self.total_volume - self.remaining_volume))
            
            if volume_to_execute >= 1:
                return {
                    'type': 'market',
                    'side': 'sell',
                    'volume': min(int(volume_to_execute), self.remaining_volume)
                }
                
        elif self.strategy == 'smart':
            # Smart strategy - use limit orders when spread is wide
            if depth['best_bid'] and depth['best_ask']:
                spread = depth['best_ask'] - depth['best_bid']
                
                if spread > 0.03:  # Wide spread - use limit order
                    return {
                        'type': 'limit',
                        'side': 'ask',
                        'price': depth['best_ask'] - 0.01,  # Improve best ask
                        'volume': min(20, self.remaining_volume)
                    }
                else:
                    # Narrow spread - use market order for small amounts
                    return {
                        'type': 'market',
                        'side': 'sell',
                        'volume': min(10, self.remaining_volume)
                    }
        
        return {'type': 'none'}
    
    def execute_action(self, action: Dict, lob: SimpleLOB) -> Optional[Dict]:
        """Execute the chosen action"""
        if action['type'] == 'market':
            trade = lob.market_order(self.agent_id, action['side'], action['volume'])
            self.remaining_volume -= trade['volume']
            self.executed_trades.append(trade)
            return trade
            
        elif action['type'] == 'limit':
            order_id = lob.add_order(self.agent_id, action['side'], action['price'], action['volume'])
            return {'type': 'limit_placed', 'order_id': order_id}
            
        return None
    
    def get_performance_metrics(self, initial_mid_price: float) -> Dict:
        """Calculate performance metrics"""
        if not self.executed_trades:
            return {
                'total_executed': 0,
                'avg_price': 0,
                'implementation_shortfall': 0,
                'fill_rate': 0
            }
        
        total_executed = sum(trade['volume'] for trade in self.executed_trades)
        total_cash = sum(trade['avg_price'] * trade['volume'] for trade in self.executed_trades)
        avg_price = total_cash / total_executed if total_executed > 0 else 0
        
        # Implementation shortfall vs initial mid price
        implementation_shortfall = (initial_mid_price - avg_price) * total_executed
        fill_rate = total_executed / self.total_volume
        
        return {
            'total_executed': total_executed,
            'avg_price': avg_price,
            'implementation_shortfall': implementation_shortfall,
            'fill_rate': fill_rate,
            'remaining_volume': self.remaining_volume
        }
