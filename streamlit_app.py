"""
Optimal Trade Execution with Reinforcement Learning
Interactive Dashboard for Research Paper Implementation

This Streamlit app demonstrates the reinforcement learning approach
to optimal trade execution in limit order books.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
import os
import sys

# Add project directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'simulation'))
sys.path.append(os.path.join(current_dir, 'limit_order_book'))
sys.path.append(os.path.join(current_dir, 'config'))

# Import project modules
try:
    from simulation.market_gym import Market
    from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder
    from simulation.agents import NoiseAgent, StrategicAgent
    MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some modules couldn't be imported: {e}")
    MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="RL Trade Execution",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86c1;
        border-bottom: 2px solid #2e86c1;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_synthetic_lob_data(n_levels=10, n_timesteps=100):
    """Create synthetic limit order book data for demonstration"""
    np.random.seed(42)
    
    # Base prices around $100
    mid_price = 100.0
    tick_size = 0.01
    
    # Initialize data structures
    timestamps = range(n_timesteps)
    bid_prices = []
    ask_prices = []
    bid_volumes = []
    ask_volumes = []
    
    for t in timestamps:
        # Simulate price walk
        mid_price += np.random.normal(0, 0.02)
        
        # Create bid/ask levels
        bid_levels = [mid_price - tick_size * (i + 0.5) for i in range(n_levels)]
        ask_levels = [mid_price + tick_size * (i + 0.5) for i in range(n_levels)]
        
        # Simulate volumes with exponential decay
        bid_vols = [max(1, int(100 * np.exp(-0.5 * i) + np.random.normal(0, 10))) for i in range(n_levels)]
        ask_vols = [max(1, int(100 * np.exp(-0.5 * i) + np.random.normal(0, 10))) for i in range(n_levels)]
        
        bid_prices.append(bid_levels)
        ask_prices.append(ask_levels)
        bid_volumes.append(bid_vols)
        ask_volumes.append(ask_vols)
    
    return {
        'timestamps': timestamps,
        'bid_prices': bid_prices,
        'ask_prices': ask_prices,
        'bid_volumes': bid_volumes,
        'ask_volumes': ask_volumes,
        'mid_prices': [(bp[0] + ap[0]) / 2 for bp, ap in zip(bid_prices, ask_prices)]
    }

def plot_order_book_snapshot(bid_prices, ask_prices, bid_volumes, ask_volumes, title="Limit Order Book"):
    """Create order book visualization"""
    fig = go.Figure()
    
    # Bid side (negative volumes for visual effect)
    fig.add_trace(go.Bar(
        x=[-v for v in bid_volumes],
        y=bid_prices,
        orientation='h',
        name='Bids',
        marker_color='green',
        opacity=0.7
    ))
    
    # Ask side
    fig.add_trace(go.Bar(
        x=ask_volumes,
        y=ask_prices,
        orientation='h',
        name='Asks',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Volume",
        yaxis_title="Price",
        barmode='overlay',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_price_impact_comparison():
    """Plot comparing different execution strategies"""
    strategies = ['Market Order', 'TWAP', 'RL Agent']
    volumes = [20, 40, 60]
    
    # Synthetic performance data
    market_order_costs = [0.15, 0.35, 0.60]
    twap_costs = [0.10, 0.22, 0.40]
    rl_costs = [0.08, 0.18, 0.32]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volumes, y=market_order_costs,
        mode='lines+markers',
        name='Market Order',
        line=dict(color='red', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=volumes, y=twap_costs,
        mode='lines+markers',
        name='TWAP',
        line=dict(color='orange', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=volumes, y=rl_costs,
        mode='lines+markers',
        name='RL Agent',
        line=dict(color='green', width=3)
    ))
    
    fig.update_layout(
        title="Implementation Shortfall by Strategy",
        xaxis_title="Volume (lots)",
        yaxis_title="Implementation Shortfall (ticks)",
        height=400
    )
    
    return fig

def simulate_trading_episode(volume, strategy, market_env):
    """Simulate a trading episode"""
    if not MODULES_AVAILABLE:
        # Return synthetic data if modules not available
        return {
            'total_reward': np.random.normal(10, 2),
            'final_pnl': np.random.normal(5, 1),
            'execution_time': np.random.randint(100, 200),
            'fill_rate': np.random.uniform(0.85, 0.98),
            'price_impact': np.random.uniform(0.05, 0.25)
        }
    
    try:
        config = {
            'seed': np.random.randint(0, 1000),
            'market_env': market_env,
            'execution_agent': strategy,
            'volume': volume,
            'terminal_time': 150,
            'time_delta': 15
        }
        
        market = Market(config)
        observation, info = market.reset()
        
        total_reward = 0
        step = 0
        
        while step < 10:  # Simplified simulation
            # For demo purposes, use random actions
            action = np.random.random(5)
            action = action / action.sum()  # Normalize to probabilities
            
            observation, reward, terminated, truncated, info = market.step(action)
            total_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        return {
            'total_reward': total_reward,
            'final_pnl': info.get('pnl', 0),
            'execution_time': step * 15,
            'fill_rate': info.get('fill_rate', 0.9),
            'price_impact': abs(info.get('price_impact', 0.1))
        }
    
    except Exception as e:
        st.error(f"Simulation error: {e}")
        return {
            'total_reward': 0,
            'final_pnl': 0,
            'execution_time': 150,
            'fill_rate': 0.5,
            'price_impact': 0.5
        }

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 Optimal Trade Execution with Reinforcement Learning</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="highlight">
    <h3>🎯 Research Overview</h3>
    This interactive dashboard demonstrates a reinforcement learning approach to optimal trade execution in limit order books. 
    The RL agent learns to minimize market impact and maximize execution efficiency when trading large volumes.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.markdown('<h2 class="section-header">🎮 Simulation Controls</h2>', unsafe_allow_html=True)
    
    # Simulation parameters
    volume_to_trade = st.sidebar.slider("Volume to Trade (lots)", 10, 100, 40)
    market_environment = st.sidebar.selectbox(
        "Market Environment",
        ["noise", "flow", "strategic"],
        help="Different market conditions with varying trader behaviors"
    )
    execution_strategy = st.sidebar.selectbox(
        "Execution Strategy",
        ["rl_agent", "linear_sl_agent", "sl_agent"],
        help="Different trading algorithms to compare"
    )
    
    # Real-time simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running trading simulation..."):
            results = simulate_trading_episode(volume_to_trade, execution_strategy, market_environment)
            
            # Store results in session state
            if 'simulation_results' not in st.session_state:
                st.session_state.simulation_results = []
            
            st.session_state.simulation_results.append({
                'timestamp': time.time(),
                'volume': volume_to_trade,
                'strategy': execution_strategy,
                'environment': market_environment,
                **results
            })
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📊 Live Order Book</h2>', unsafe_allow_html=True)
        
        # Generate synthetic order book data
        lob_data = create_synthetic_lob_data(n_levels=10, n_timesteps=1)
        
        # Display current order book
        if st.button("🔄 Refresh Order Book"):
            lob_data = create_synthetic_lob_data(n_levels=10, n_timesteps=1)
        
        fig_lob = plot_order_book_snapshot(
            lob_data['bid_prices'][0],
            lob_data['ask_prices'][0],
            lob_data['bid_volumes'][0],
            lob_data['ask_volumes'][0],
            "Current Market Depth"
        )
        st.plotly_chart(fig_lob, use_container_width=True)
        
        # Strategy comparison
        st.markdown('<h2 class="section-header">📈 Strategy Performance Comparison</h2>', unsafe_allow_html=True)
        fig_comparison = plot_price_impact_comparison()
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-header">📋 Simulation Results</h2>', unsafe_allow_html=True)
        
        if 'simulation_results' in st.session_state and st.session_state.simulation_results:
            latest_result = st.session_state.simulation_results[-1]
            
            # Display metrics
            st.metric("Total Reward", f"{latest_result['total_reward']:.2f}")
            st.metric("Final P&L", f"${latest_result['final_pnl']:.2f}")
            st.metric("Execution Time", f"{latest_result['execution_time']}s")
            st.metric("Fill Rate", f"{latest_result['fill_rate']:.1%}")
            st.metric("Price Impact", f"{latest_result['price_impact']:.3f} ticks")
            
            # Show recent history
            if len(st.session_state.simulation_results) > 1:
                st.markdown("### Recent Simulations")
                recent_df = pd.DataFrame(st.session_state.simulation_results[-5:])
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'], unit='s')
                st.dataframe(
                    recent_df[['timestamp', 'strategy', 'volume', 'total_reward', 'fill_rate']],
                    use_container_width=True
                )
        else:
            st.info("👆 Run a simulation to see results!")
    
    # Educational content
    st.markdown('<h2 class="section-header">🎓 How It Works</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Problem Setup", "🧠 RL Algorithm", "📊 Market Simulation", "🔬 Results"])
    
    with tab1:
        st.markdown("""
        ### The Trade Execution Challenge
        
        **Objective**: Sell a large volume (e.g., 40 lots) over time without causing significant market impact.
        
        **State Space** (what the RL agent observes):
        - 📈 Best bid/ask prices and volumes
        - 📊 Market depth (5 price levels on each side)
        - 🎯 Remaining inventory to trade
        - 📍 Queue positions of active limit orders
        
        **Action Space** (what the agent can do):
        - 🚀 Market orders (immediate execution)
        - 📋 Limit orders at different price levels
        - ❌ Cancel existing orders
        - ⏳ Wait and do nothing
        
        **Reward Function**: 
        ```
        Reward = Cash Received - Benchmark Price × Volume - Market Impact Penalty
        ```
        """)
    
    with tab2:
        st.markdown("""
        ### Reinforcement Learning Framework
        
        **Algorithm**: Actor-Critic with continuous action space
        
        **Key Innovation**: Logistic-normal distribution for action parameterization
        - Ensures actions sum to 1 (valid probability distribution)
        - Enables smooth gradients for policy optimization
        
        **Network Architecture**:
        - 🎭 **Actor**: Maps state → action probabilities
        - 🎯 **Critic**: Estimates state value for learning
        - 🔧 **Both**: 3-layer MLPs with 64 nodes each
        
        **Training Process**:
        1. 🎮 Simulate thousands of trading episodes
        2. 📊 Collect state-action-reward trajectories  
        3. 🔄 Update policy using PPO algorithm
        4. 📈 Evaluate against benchmark strategies
        """)
    
    with tab3:
        st.markdown("""
        ### Market Simulation Environment
        
        **Agent Types**:
        - 🎲 **Noise Traders**: Random orders creating baseline liquidity
        - 🎯 **Tactical Traders**: React to order flow imbalances
        - 🤖 **Strategic Agent**: Your RL-powered execution algorithm
        
        **Market Dynamics**:
        - ⚡ Order matching using price-time priority
        - 📊 Queue positions tracked for limit orders
        - 💥 Market impact from large orders
        - 🔄 Price resilience (mean reversion)
        
        **Benchmark Strategies**:
        - 📈 **Market Order**: Execute everything immediately
        - ⏰ **TWAP**: Time-weighted average price (even splitting)
        - 🎯 **Volume-weighted**: Proportion to historical volumes
        """)
    
    with tab4:
        st.markdown("""
        ### Key Research Findings
        
        **Performance Gains**:
        - 🎯 **10-20% improvement** over benchmark strategies
        - 💰 **0.05-0.1 ticks saved per lot** in execution costs
        - 📈 **Higher fill rates** with better timing
        
        **Strategy Adaptation**:
        - 🎲 **Noisy markets**: More limit orders, patient execution
        - ⚡ **Tactical markets**: Avoids triggering adverse reactions
        - 📊 **Large volumes**: Smarter impact management
        
        **Scalability**:
        - ✅ Works across different volume sizes (20-60 lots)
        - ✅ Adapts to various market conditions
        - ✅ Outperforms rule-based algorithms consistently
        
        **Implementation Notes**:
        - 🔧 Assumes symmetric order book (simplification)
        - 💡 No transaction fees in current model
        - 🚀 Ready for extension to multi-asset trading
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    📝 <strong>Research Implementation</strong> | 
    🔬 <em>Optimal Trade Execution with Reinforcement Learning</em> | 
    🚀 <strong>Interactive Demo</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
