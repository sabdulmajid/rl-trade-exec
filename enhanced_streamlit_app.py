"""
Enhanced Streamlit App for RL Trade Execution Demo
Uses simplified components for reliable web deployment
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from simple_lob_simulator import SimpleMarketSimulator, SimpleExecutionAgent

# Page configuration
st.set_page_config(
    page_title="RL Trade Execution Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def plot_order_book_depth(depth_data, title="Market Depth"):
    """Create order book depth visualization"""
    fig = go.Figure()
    
    if depth_data['bids']:
        bid_prices = [level['price'] for level in depth_data['bids']]
        bid_volumes = [-level['volume'] for level in depth_data['bids']]  # Negative for visual effect
        
        fig.add_trace(go.Bar(
            x=bid_volumes,
            y=bid_prices,
            orientation='h',
            name='Bids',
            marker_color='rgba(0, 128, 0, 0.7)',
            hovertemplate='Price: $%{y:.2f}<br>Volume: %{x}<extra></extra>'
        ))
    
    if depth_data['asks']:
        ask_prices = [level['price'] for level in depth_data['asks']]
        ask_volumes = [level['volume'] for level in depth_data['asks']]
        
        fig.add_trace(go.Bar(
            x=ask_volumes,
            y=ask_prices,
            orientation='h',
            name='Asks',
            marker_color='rgba(255, 0, 0, 0.7)',
            hovertemplate='Price: $%{y:.2f}<br>Volume: %{x}<extra></extra>'
        ))
    
    # Add mid price line
    if depth_data['best_bid'] and depth_data['best_ask']:
        mid_price = (depth_data['best_bid'] + depth_data['best_ask']) / 2
        fig.add_hline(
            y=mid_price, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Mid: ${mid_price:.2f}"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Volume (negative=bids, positive=asks)",
        yaxis_title="Price ($)",
        barmode='overlay',
        height=400,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def plot_execution_progress(agents_data):
    """Plot execution progress over time"""
    fig = go.Figure()
    
    colors = {'market': 'red', 'twap': 'orange', 'smart': 'green'}
    
    for strategy, data in agents_data.items():
        if data['executed_volumes']:
            fig.add_trace(go.Scatter(
                x=list(range(len(data['executed_volumes']))),
                y=data['executed_volumes'],
                mode='lines+markers',
                name=f'{strategy.upper()} Strategy',
                line=dict(color=colors.get(strategy, 'blue'), width=3)
            ))
    
    fig.update_layout(
        title="Execution Progress Over Time",
        xaxis_title="Time Steps",
        yaxis_title="Cumulative Executed Volume",
        height=400
    )
    
    return fig

def plot_price_comparison(agents_data):
    """Plot average execution prices"""
    strategies = []
    avg_prices = []
    
    for strategy, data in agents_data.items():
        if data['trades']:
            total_volume = sum(trade['volume'] for trade in data['trades'])
            total_cash = sum(trade['avg_price'] * trade['volume'] for trade in data['trades'])
            avg_price = total_cash / total_volume if total_volume > 0 else 0
            strategies.append(strategy.upper())
            avg_prices.append(avg_price)
    
    if strategies:
        fig = go.Figure(data=go.Bar(
            x=strategies,
            y=avg_prices,
            marker_color=['red', 'orange', 'green'][:len(strategies)]
        ))
        
        fig.update_layout(
            title="Average Execution Price by Strategy",
            xaxis_title="Strategy",
            yaxis_title="Average Price ($)",
            height=400
        )
        
        return fig
    
    return None

def run_trading_simulation(volume, time_limit=50):
    """Run a complete trading simulation"""
    # Initialize market simulator
    simulator = SimpleMarketSimulator(initial_mid_price=100.0)
    initial_mid = simulator.mid_price
    
    # Create execution agents with different strategies
    agents = {
        'market': SimpleExecutionAgent('market_agent', volume, 'market'),
        'twap': SimpleExecutionAgent('twap_agent', volume, 'twap'),
        'smart': SimpleExecutionAgent('smart_agent', volume, 'smart')
    }
    
    # Data tracking
    market_states = []
    agents_data = {strategy: {
        'executed_volumes': [],
        'remaining_volumes': [],
        'trades': [],
        'final_metrics': {}
    } for strategy in agents.keys()}
    
    # Run simulation
    for step in range(time_limit):
        # Advance market
        market_state = simulator.step()
        market_states.append(market_state)
        
        # Each agent takes action
        for strategy, agent in agents.items():
            if agent.remaining_volume > 0:
                action = agent.get_action(market_state, simulator.lob.time)
                result = agent.execute_action(action, simulator.lob)
                
                if result and result.get('volume', 0) > 0:
                    agents_data[strategy]['trades'].append(result)
            
            # Track progress
            executed = agent.total_volume - agent.remaining_volume
            agents_data[strategy]['executed_volumes'].append(executed)
            agents_data[strategy]['remaining_volumes'].append(agent.remaining_volume)
    
    # Calculate final metrics
    for strategy, agent in agents.items():
        agents_data[strategy]['final_metrics'] = agent.get_performance_metrics(initial_mid)
    
    return {
        'market_states': market_states,
        'agents_data': agents_data,
        'initial_mid_price': initial_mid,
        'final_market_state': market_states[-1] if market_states else None
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Trade Execution: Live Demo</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="metric-container">
    <h3>Interactive Trading Simulation</h3>
    Experience how different execution strategies perform in a live limit order book environment. 
    Watch as Market Orders, TWAP, and Smart strategies compete to execute your trade efficiently.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## Simulation Settings")
    
    volume_to_trade = st.sidebar.slider(
        "Volume to Execute (lots)", 
        min_value=10, 
        max_value=200, 
        value=50,
        help="Total number of lots to trade"
    )
    
    simulation_speed = st.sidebar.selectbox(
        "Simulation Speed",
        options=["Fast", "Medium", "Slow"],
        index=1,
        help="How quickly to run the simulation"
    )
    
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh Market", 
        value=False,
        help="Automatically update the order book"
    )
    
    # Main simulation controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Start New Simulation", type="primary"):
            with st.spinner("Running trading simulation..."):
                sim_results = run_trading_simulation(volume_to_trade)
                st.session_state['sim_results'] = sim_results
                st.session_state['simulation_complete'] = True
                st.success("Simulation completed!")
    
    with col2:
        if st.button("Refresh Market"):
            # Generate new market state
            if 'market_sim' not in st.session_state:
                st.session_state['market_sim'] = SimpleMarketSimulator()
            else:
                st.session_state['market_sim'].step()
    
    # Auto-refresh functionality
    if auto_refresh:
        if 'market_sim' not in st.session_state:
            st.session_state['market_sim'] = SimpleMarketSimulator()
        
        # Simulate market evolution
        market_state = st.session_state['market_sim'].step()
        time.sleep(0.1)  # Small delay for visual effect
        st.rerun()
    
    # Display current market state
    if 'market_sim' in st.session_state:
        current_market = st.session_state['market_sim']
        depth = current_market.lob.get_market_depth()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Live Order Book")
            fig_depth = plot_order_book_depth(depth, "Current Market Depth")
            st.plotly_chart(fig_depth, use_container_width=True)
        
        with col2:
            st.markdown("### Market Info")
            if depth['best_bid'] and depth['best_ask']:
                mid_price = (depth['best_bid'] + depth['best_ask']) / 2
                spread = depth['best_ask'] - depth['best_bid']
                
                st.metric("Mid Price", f"${mid_price:.2f}")
                st.metric("Best Bid", f"${depth['best_bid']:.2f}")
                st.metric("Best Ask", f"${depth['best_ask']:.2f}")
                st.metric("Spread", f"${spread:.3f}")
                
                # Spread condition indicator
                if spread > 0.05:
                    st.markdown('<div class="metric-container danger-metric">Wide Spread - Good for Limit Orders</div>', unsafe_allow_html=True)
                elif spread < 0.02:
                    st.markdown('<div class="metric-container success-metric">Tight Spread - Market Orders OK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-container warning-metric">Normal Spread</div>', unsafe_allow_html=True)
    
    # Simulation results
    if 'sim_results' in st.session_state and st.session_state.get('simulation_complete', False):
        st.markdown("---")
        st.markdown("## Simulation Results")
        
        sim_data = st.session_state['sim_results']
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        strategies = ['market', 'twap', 'smart']
        strategy_names = ['Market Order', 'TWAP', 'Smart Strategy']
        
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
            metrics = sim_data['agents_data'][strategy]['final_metrics']
            
            with [col1, col2, col3][i]:
                st.markdown(f"### {name}")
                
                fill_rate = metrics.get('fill_rate', 0)
                avg_price = metrics.get('avg_price', 0)
                shortfall = metrics.get('implementation_shortfall', 0)
                
                st.metric("Fill Rate", f"{fill_rate:.1%}")
                st.metric("Avg Price", f"${avg_price:.2f}")
                st.metric("P&L", f"${shortfall:.2f}")
                
                # Performance indicator
                if fill_rate > 0.95:
                    st.markdown('<div class="metric-container success-metric">Excellent Execution</div>', unsafe_allow_html=True)
                elif fill_rate > 0.8:
                    st.markdown('<div class="metric-container warning-metric">Good Execution</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-container danger-metric">Poor Execution</div>', unsafe_allow_html=True)
        
        # Execution progress chart
        st.markdown("### Execution Progress")
        fig_progress = plot_execution_progress(sim_data['agents_data'])
        st.plotly_chart(fig_progress, use_container_width=True)
        
        # Price comparison
        st.markdown("### Price Performance")
        fig_prices = plot_price_comparison(sim_data['agents_data'])
        if fig_prices:
            st.plotly_chart(fig_prices, use_container_width=True)
        
        # Performance summary table
        st.markdown("### Detailed Performance Summary")
        
        summary_data = []
        for strategy in strategies:
            metrics = sim_data['agents_data'][strategy]['final_metrics']
            summary_data.append({
                'Strategy': strategy.upper(),
                'Fill Rate': f"{metrics.get('fill_rate', 0):.1%}",
                'Average Price': f"${metrics.get('avg_price', 0):.2f}",
                'Total Executed': metrics.get('total_executed', 0),
                'Remaining': metrics.get('remaining_volume', 0),
                'Implementation Shortfall': f"${metrics.get('implementation_shortfall', 0):.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Educational content
    st.markdown("---")
    st.markdown("## 🎓 Understanding the Strategies")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Orders", "⏰ TWAP", "🧠 Smart Strategy", "🤖 RL Approach"])
    
    with tab1:
        st.markdown("""
        ### Market Order Strategy
        
        **Approach**: Execute the entire volume immediately using market orders.
        
        **Pros**:
        - Guaranteed execution
        - Simple to implement
        - No timing risk
        
        **Cons**:
        - High market impact
        - Poor prices in illiquid markets
        - No optimization for cost
        
        **Best Used When**: Need immediate execution regardless of cost.
        """)
    
    with tab2:
        st.markdown("""
        ### Time-Weighted Average Price (TWAP)
        
        **Approach**: Split the order into smaller pieces executed at regular intervals.
        
        **Pros**:
        - Reduces market impact
        - Spreads execution risk over time
        - Predictable execution pattern
        
        **Cons**:
        - Timing risk if market moves against you
        - Not adaptive to market conditions
        - May not complete if market becomes illiquid
        
        **Best Used When**: Markets are relatively stable and liquid.
        """)
    
    with tab3:
        st.markdown("""
        ### Smart Strategy (Rule-Based)
        
        **Approach**: Adapt execution method based on current market conditions.
        
        **Logic**:
        - **Wide Spread**: Use limit orders to capture spread
        - **Tight Spread**: Use market orders for quick execution
        - **Volume Targeting**: Adjust order sizes based on market depth
        
        **Pros**:
        - Adapts to market conditions
        - Better than fixed strategies
        - Balances cost and execution certainty
        
        **Cons**:
        - Still rule-based (limited adaptability)
        - Doesn't learn from experience
        - May miss complex market patterns
        """)
    
    with tab4:
        st.markdown("""
        ### Reinforcement Learning Approach
        
        **How It Works**: An AI agent learns optimal execution through trial and error.
        
        **Key Advantages**:
        - **Learns Complex Patterns**: Discovers non-obvious strategies
        - **Adapts to Any Market**: Handles noise, flow, and strategic environments
        - **Optimizes for Objectives**: Directly maximizes reward function
        - **Continuous Improvement**: Gets better with more data
        
        **State Space** (what the agent sees):
        ```
        • Best bid/ask prices and volumes
        • Market depth (multiple price levels)
        • Remaining inventory to execute
        • Queue positions of pending orders
        • Recent price movements and trends
        ```
        
        **Action Space** (what the agent can do):
        ```
        • Market order (% of remaining volume)
        • Limit orders at different price levels
        • Cancel existing orders
        • Wait for better conditions
        ```
        
        **Training Process**:
        1. Simulate thousands of trading scenarios
        2. Try different actions and observe results
        3. Update policy to maximize long-term rewards
        4. Evaluate against benchmark strategies
        
        **Research Results**: 10-20% improvement over traditional methods!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <strong>Research Implementation</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
