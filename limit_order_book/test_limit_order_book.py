from limit_order_book import LimitOrderBook, LimitOrder, MarketOrder, Cancellation, Modification, CancellationByPriceVolume
import unittest
import numpy as np

class TestOrderBook(unittest.TestCase):

    def test_volume_only_mode(self):
        ## 1 
        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10, only_volumes=True)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0)) #0 
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1)) #1
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 3)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 4)) #4
        orders.append(LimitOrder('smart_agent', 'ask', 102, 2, 5)) #5
        [LOB.process_order(order) for order in orders]
        order = CancellationByPriceVolume(agent_id='noise_agent', price=102, volume=4, side='ask', time=6)
        msg = LOB.process_order(order)
        assert LOB.price_volume_map['ask'][102] == 5 + 2 + 2 - 4
        order = CancellationByPriceVolume(agent_id='noise_agent', price=102, volume=4, side='ask', time=7)
        mssg = LOB.process_order(order)
        assert LOB.price_volume_map['ask'][102] == 5 + 2 + 2 - 4 - 4
        order = CancellationByPriceVolume(agent_id='noise_agent', price=102, volume=4, side='ask', time=8)
        mssg = LOB.process_order(order)
        assert 102 not in LOB.price_volume_map['ask']
        ## 2 
        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10, only_volumes=True)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0)) #0 
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1)) #1
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 3)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 4)) #4
        orders.append(LimitOrder('smart_agent', 'ask', 102, 2, 5)) #5
        orders.append(LimitOrder('noise_agent', 'ask', 105, 1, 6)) #5
        orders.append(LimitOrder('smart_agent', 'ask', 110, 2, 7)) #5
        [LOB.process_order(order) for order in orders]
        order = MarketOrder('noise_agent', 'ask', 5+5+2+2+1+1, 8)
        LOB.process_order(order)
        assert 101 not in LOB.price_volume_map['ask']
        assert 102 not in LOB.price_volume_map['ask']
        assert 105 not in LOB.price_volume_map['ask']
        assert LOB.price_volume_map['ask'][110] == 1 
        ## 3
        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10, only_volumes=True)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0)) #0 
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1)) #1
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 3)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 4)) #4
        orders.append(LimitOrder('smart_agent', 'ask', 102, 2, 5)) #5
        orders.append(LimitOrder('noise_agent', 'ask', 105, 1, 6)) #5
        orders.append(LimitOrder('smart_agent', 'ask', 110, 2, 7)) #5
        [LOB.process_order(order) for order in orders]
        assert LOB.price_volume_map['bid'][100] == 5
        assert LOB.price_volume_map['ask'][101] == 5
        assert LOB.price_volume_map['ask'][102] == 5+2+2
        assert LOB.price_volume_map['ask'][105] == 1
        assert LOB.price_volume_map['ask'][110] == 2
        return None 


    def test_cancellation_by_volume(self, only_volumes=False):
        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10, only_volumes=only_volumes)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 3))
        orders.append(LimitOrder('smart_agent', 'ask', 102, 2, 4))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 5))
        [LOB.process_order(order) for order in orders]
        v1 = np.sum(LOB.data.ask_volumes[-1][1]) 
        order = CancellationByPriceVolume(agent_id='noise_agent', price=102, volume=4, side='ask', time=6)
        msg = LOB.process_order(order)
        v2 = np.sum(LOB.data.ask_volumes[-1][1])
        assert LOB.data.ask_volumes[-1][1] == 5 + 2 + 2 - 4 
        assert LOB.price_volume_map['ask'][102] == 5 + 2 + 2 - 4
        assert LOB.order_map[LOB.price_map['ask'][102][-1]].agent_id == 'smart_agent'
        assert msg.filled_volume == v1-v2


        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0))
        orders.append(LimitOrder('noise_agent', 'ask', 100, 5, 1))
        orders.append(LimitOrder('smart_agent', 'ask', 100, 2, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 100, 2, 3))
        [LOB.process_order(order) for order in orders]
        v1 = np.sum(LOB.data.ask_volumes[-1][0])
        order = CancellationByPriceVolume(agent_id='noise_agent', price=100, volume=10, side='ask', time=4)
        msg = LOB.process_order(order)
        v2 = np.sum(LOB.data.ask_volumes[-1][0])
        assert LOB.data.ask_volumes[-1][0] == 2
        assert LOB.price_volume_map['ask'][100] == 2
        assert msg.filled_volume == v1-v2
        assert msg.partial_fill == True
        assert 1 not in LOB.order_map
        assert 3 not in LOB.order_map 


        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10)
        orders = []
        orders.append(LimitOrder('smart_agent', 'bid', 99, 10, 0))
        orders.append(LimitOrder('smart_agent', 'bid', 99, 1, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 2, 2))
        orders.append(CancellationByPriceVolume(agent_id='noise_agent', price=99, volume=6, side='bid', time=3))
        [LOB.process_order(order) for order in orders]

        assert LOB.data.bid_volumes[-1][0] == 10 + 1 + 2 - 2
        assert LOB.price_volume_map['bid'][99] == 10 + 1 + 2 - 2
        assert LOB.order_map[LOB.price_map['bid'][99][-1]].agent_id == 'smart_agent'


        return None



    def test_logging(self, only_volumes=True):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'], level=3,  only_volumes=only_volumes)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0)) #1
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10, 3)) #4
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 4)) #5
        [LOB.process_order(order) for order in orders]  
        assert len(LOB.data.orders) == 5
        assert len(LOB.data.best_ask_prices) == 5
        assert LOB.data.best_ask_prices == [np.nan, np.nan, 101, 101, 101]
        assert LOB.data.best_bid_prices == [99, 100, 100, 100, 100]
        assert np.all(LOB.data.ask_volumes[-1] == np.array([5, 15, 0]))
        assert np.all(LOB.data.bid_volumes[-1] == np.array([5, 10, 0]))
        assert np.all(LOB.data.ask_prices[-1] == np.array([101, 102, 103]))
        assert np.all(LOB.data.bid_prices[-1] == np.array([100, 99, 98]))
        # 
        order = MarketOrder('smart_agent', 'ask', 5, 5)
        LOB.process_order(order)
        assert np.all(LOB.data.ask_volumes[-1] == np.array([0, 15, 0]))
        return None 



    def test_limit_order_insertion(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 4))
        msg = [LOB.process_order(order) for order in orders]
        assert LOB.price_volume_map['bid'][99] == 10
        assert LOB.price_volume_map['bid'][100] == 5
        assert LOB.price_volume_map['ask'][101] == 5
        assert LOB.price_volume_map['ask'][102] == 15
        # [print(m) for m in msg]
        return None


    def test_market_order_insertion(self):
        LOB = LimitOrderBook(list_of_agents=['smart_agent', 'noise_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 4))
        orders.append(MarketOrder('smart_agent', 'bid', 10, 5))
        msg = [LOB.process_order(order) for order in orders]
        # [print(m) for m in msg]
        assert LOB.price_volume_map['bid'][99] == 5
        assert 100 not in LOB.price_volume_map['bid']
        return None


    def test_cancellation_0(self):
        LOB = LimitOrderBook(list_of_agents = ['smart_agent', 'noise_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5, 4))
        orders.append(Cancellation(order_id=3, agent_id='noise_agent', time=5))
        msg = [LOB.process_order(order) for order in orders]
        # [print(m.msg) for m in msg]        
        assert LOB.price_volume_map['ask'][102] == 5
        order = Cancellation(order_id=4, agent_id='noise_agent', time=6)
        msg = LOB.process_order(order)
        assert 102 not in LOB.price_volume_map['ask']
        return None
    

    def test_limit_order_fill(self, partial_fill=False):
        """
        - add some limit orders to the book 
        - add one limit order by smart agent
        - check if the order is filled
        """
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        # add noise agent orders 
        orders = []
        # bid 
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1, 0))
        # ask 
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 1, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 4))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 105, 3, 6))
        # total ask side = 10
        [LOB.process_order(order) for order in orders]
        # add smart agent order
        if partial_fill:
            msg = LimitOrder('smart_agent', 'ask', 104, 2, 7)
            msg = LOB.process_order(msg)
            order_id = msg.order_id
            order = MarketOrder('noise_agent', 'ask', 8, 8)
            msg = LOB.process_order(order)
            assert msg.passive_fills['smart_agent'][0].partial_fill == True
            assert msg.passive_fills['smart_agent'][0].filled_volume == 1 
            assert LOB.order_map[order_id].volume == 1
            assert order_id in LOB.order_map
            assert order_id in LOB.order_map_by_agent['smart_agent']
        else:
            msg = LimitOrder('smart_agent', 'ask', 104, 2, 7)
            msg = LOB.process_order(msg)
            assert msg.agent_id == 'smart_agent'
            order_id = msg.order_id
            order = MarketOrder('noise_agent', 'ask', 10, 8)
            msg = LOB.process_order(order)
            assert msg.passive_fills['smart_agent'][0].partial_fill == False
            assert msg.passive_fills['smart_agent'][0].filled_volume == 2 
            assert order_id not in LOB.order_map
            assert order_id not in LOB.order_map_by_agent['smart_agent']
        return None 
    

    def test_fill_time(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'ask', 101, 10, 0))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 6, 1))
        orders.append(LimitOrder('smart_agent', 'ask', 103, 2, 2))
        [LOB.process_order(order) for order in orders]
        orders = []
        filled = False
        t = 0 
        while not filled:
            order = MarketOrder('noise_agent', 'ask', 2, 3)
            msg = LOB.process_order(order)
            if 'smart_agent' in msg.passive_fills:
                filled = True
                assert msg.passive_fills['smart_agent'][0].filled_volume == 2
                assert t == 5 + 3 
            t += 1
        # new book 
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1, 0))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 1, 1))
        orders.append(LimitOrder('smart_agent', 'ask', 104, 1, 2))
        [LOB.process_order(order) for order in orders]
        orders = []
        filled = False
        t = 0 
        n = 3
        while not filled:
            order = MarketOrder('noise_agent', 'ask', 1, n)
            msg = LOB.process_order(order)
            if 'smart_agent' in msg.passive_fills:
                filled = True
                assert msg.passive_fills['smart_agent'][0].filled_volume == 1
                assert 104 not in LOB.price_map['ask']
                assert 2 not in LOB.order_map
                assert 2 not in LOB.order_map_by_agent['smart_agent']
                assert t == 2
            t += 1
            n += 1
        return None
    
    def test_passive_fill(self):
        # 1
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 10, 0))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 10, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 6, 2))
        orders.append(LimitOrder('smart_agent', 'ask', 109, 2, 3))
        orders.append(LimitOrder('smart_agent', 'ask', 109, 3, 4))
        orders.append(LimitOrder('smart_agent', 'ask', 110, 2, 5))
        [LOB.process_order(order) for order in orders]
        order = MarketOrder('noise_agent', 'ask', 10+6+2+1, 6)
        msg = LOB.process_order(order)
        assert 'smart_agent' in msg.passive_fills
        assert len(msg.passive_fills['smart_agent']) == 2
        assert msg.passive_fills['smart_agent'][0].filled_volume == 2
        assert msg.passive_fills['smart_agent'][0].partial_fill == False
        id = msg.passive_fills['smart_agent'][0].order.order_id
        assert id not in LOB.order_map
        assert msg.passive_fills['smart_agent'][1].filled_volume == 1
        assert msg.passive_fills['smart_agent'][1].partial_fill == True
        id = msg.passive_fills['smart_agent'][1].order.order_id
        assert LOB.order_map[id].volume == 2
        # 2
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 10, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 2, 1))
        orders.append(LimitOrder('smart_agent', 'bid', 100, 2, 2))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3, 3))
        [LOB.process_order(order) for order in orders]
        order = MarketOrder('noise_agent', 'bid', 10+2+1, 4)
        msg = LOB.process_order(order)
        assert 'smart_agent' in msg.passive_fills
        assert len(msg.passive_fills['smart_agent']) == 1
        assert msg.passive_fills['smart_agent'][0].filled_volume == 1
        assert msg.passive_fills['smart_agent'][0].partial_fill == True
        id = msg.passive_fills['smart_agent'][0].order.order_id
        assert LOB.order_map[id].volume == 1
        return None



    def test_cancellation(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1, 0))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 2, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 3, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 3, 4))
        orders.append(LimitOrder('noise_agent', 'ask', 104, 2, 5))
        [LOB.process_order(order) for order in orders]
        order = LimitOrder('smart_agent', 'ask', 104, 3, 6)
        order = LOB.process_order(order)
        order_id = order.order_id
        order = Cancellation(order_id=order_id, agent_id='smart_agent', time=7)
        msg = LOB.process_order(order)
        assert msg.order.order_id not in LOB.order_map
        assert msg.order.order_id not in LOB.order_map_by_agent['smart_agent']
        assert msg.price == 104        
        assert msg.volume == 3
        assert msg.side == 'ask'
        assert msg.order.agent_id == 'smart_agent'
        assert LOB.price_volume_map['ask'][104] == 2
        return None 
    

    def test_market_order(self): 
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 3, 2))
        orders.append(LimitOrder('noise_agent', 'bid', 97, 2, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 2, 4))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 3, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2, 6))
        orders.append(LimitOrder('noise_agent', 'ask', 103, 3, 7))
        [LOB.process_order(order) for order in orders]
        market_order = MarketOrder('smart_agent', 'bid', 5, 8)
        msg = LOB.process_order(market_order)   
        assert market_order.agent_id == 'smart_agent'
        assert 'noise_agent' in msg.passive_fills
        assert msg.execution_price == 100*4+99*1
        assert 100 not in LOB.price_map['bid']
        assert len(LOB.price_map['bid'][99]) == 1
        assert LOB.order_map[LOB.price_map['bid'][99][0]] 
        assert 100 not in LOB.price_volume_map['bid']
        assert LOB.price_volume_map['bid'][99] == 2
        #####
        market_order = MarketOrder('smart_agent', 'ask', 8, 9)
        msg = LOB.process_order(market_order)
        assert 101 not in LOB.price_volume_map['ask']
        assert 102 not in LOB.price_volume_map['ask']
        assert LOB.price_volume_map['ask'][103]  == 2
        assert msg.execution_price == 101*5 + 102*2 + 103*1
        assert 101 not in LOB.price_map['ask']
        assert 102 not in LOB.price_map['ask']
        assert len(LOB.price_map['ask'][103]) == 1        
        assert LOB.order_map[LOB.price_map['ask'][103][0]].volume == 2
        assert msg.passive_fills['noise_agent'][-1].partial_fill == True
        return None 

    def test_modification(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 3, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 2, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 103, 4, 4)) #4
        [LOB.process_order(order) for order in orders]
        assert LOB.order_map[4].volume == 4
        order = Modification(order_id=4, agent_id='noise_agent', new_volume=1, time=5)
        msg = LOB.process_order(order)
        assert LOB.price_volume_map['ask'][103] == 1 
        assert LOB.order_map[4].volume == 1    
        assert msg.new_volume == 1
        assert msg.old_volume == 4
        return None

    def test_queue_position(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1, 0))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 2, 2))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 2, 3))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 3, 4))
        [LOB.process_order(order) for order in orders]
        p = LOB.find_queue_position(1)
        assert p == 1
        p = LOB.find_queue_position(3)
        assert p == 6
        return None 
        

if __name__ == '__main__':
    print("Start testing...")
    TLOB = TestOrderBook()
    TLOB.test_volume_only_mode()    
    print("test volume only mode passed...")
    TLOB.test_cancellation_by_volume()    
    print('test_cancellation_by_volume passed...')
    TLOB.test_logging(only_volumes=True)
    TLOB.test_logging(only_volumes=False)
    print('test_logging passed...')
    TLOB.test_fill_time()
    print('test_fill_time passed...')
    TLOB.test_passive_fill()
    TLOB.test_limit_order_insertion()
    TLOB.test_limit_order_fill(partial_fill=True)
    TLOB.test_limit_order_fill(partial_fill=False)
    print('##########')
    TLOB.test_cancellation_0()
    TLOB.test_cancellation()
    print('##########')
    TLOB.test_market_order()
    print('##########')
    TLOB.test_market_order_insertion()
    print('##########')
    TLOB.test_modification()
    print('##########')
    TLOB.test_queue_position()
