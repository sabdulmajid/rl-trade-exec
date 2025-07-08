from limit_order_book import LimitOrderBook, LimitOrder, MarketOrder, Cancellation
from plotting import plot_prices, plot_average_book_shape, heat_map
import matplotlib.pyplot as plt


def test_plot_prices(level=5, case=1):
    LOB = LimitOrderBook(smart_agent_id='smart_agent', noise_agent_id='noise_agent', level=level)
    orders = []
    if case == 1:
        orders.append(LimitOrder('noise_agent', 'bid', 99, 1)) #0
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1)) #1
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 102, 1)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 103, 1)) #3
        orders.append(MarketOrder('noise_agent', 'ask', 1 )) #4 
        orders.append(LimitOrder('noise_agent', 'ask', 104, 1)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 105, 1)) #3
        orders.append(LimitOrder('noise_agent', 'bid', 98, 1)) #3
    if case == 2:
        orders.append(LimitOrder('noise_agent', 'bid', 99, 1)) #0
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1)) #1
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 103, 1)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 104, 1)) #4
        orders.append(LimitOrder('noise_agent', 'ask', 105, 1)) #5 
        orders.append(LimitOrder('noise_agent', 'ask', 106, 1)) #6 
        orders.append(LimitOrder('noise_agent', 'ask', 107, 1)) #7 
        orders.append(MarketOrder('noise_agent', 'ask', 1 )) #8 
        orders.append(LimitOrder('noise_agent', 'bid', 101, 1)) #0
        orders.append(MarketOrder('noise_agent', 'ask', 1 )) #5 
        orders.append(LimitOrder('noise_agent', 'bid', 102, 1)) #0
        orders.append(MarketOrder('noise_agent', 'ask', 1 )) #5 
        orders.append(LimitOrder('noise_agent', 'bid', 103, 1)) #0
        orders.append(MarketOrder('noise_agent', 'ask', 1 )) #5 
        orders.append(LimitOrder('noise_agent', 'bid', 104, 1)) #0
        orders.append(LimitOrder('noise_agent', 'bid', 104, 9)) #0
        orders.append(LimitOrder('noise_agent', 'ask', 108, 1)) #0
        orders.append(MarketOrder('noise_agent', 'ask', 2)) #0
        orders.append(LimitOrder('noise_agent', 'ask', 108, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 108, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 108, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 110, 1))
    [LOB.process_order(order) for order in orders]
    level2, trades = LOB.log_to_df()
    plot_average_book_shape(LOB.data.bid_volumes, LOB.data.ask_volumes, level=level)
    plot_prices(level2, trades)
    heat_map(trades=trades, max_level=level, level2=level2)
    

if __name__ == "__main__":
    test_plot_prices()
    plt.show()
    print(1)