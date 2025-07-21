"""
Configuration settings for the RL Trade Execution Dashboard
Modify these settings to customize model training and evaluation
"""

# Model Architecture
MODEL_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 2,
    'activation': 'tanh',
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95
}

# Training Configuration
TRAINING_CONFIG = {
    'total_timesteps': 50000,
    'num_envs': 8,
    'batch_size': 2048,
    'mini_batch_size': 256,
    'ppo_epochs': 10,
    'clip_range': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01
}

# Environment Configuration
ENV_CONFIG = {
    'environments': ['noise', 'flow', 'strategic'],
    'volumes': [20, 40, 60, 80, 100],
    'terminal_time': 150,
    'time_delta': 15,
    'max_episode_steps': 10
}

# Evaluation Configuration
EVAL_CONFIG = {
    'n_eval_episodes': 100,
    'eval_freq': 5000,
    'strategies_to_compare': ['RL Agent', 'TWAP', 'VWAP', 'Market Order'],
    'metrics': ['mean_reward', 'implementation_shortfall', 'fill_rate', 'volatility']
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'title': "RL Trade Execution Dashboard",
    'sidebar_width': 300,
    'chart_height': 400,
    'update_frequency': 'realtime',
    'theme': 'streamlit'
}

# Data Paths
DATA_PATHS = {
    'results_dir': 'results/',
    'models_dir': 'models/',
    'logs_dir': 'tensorboard_logs/',
    'initial_shape_dir': 'initial_shape/',
    'evaluation_results': 'results/ml_evaluation_results.json',
    'training_curves': 'results/training_curves.json',
    'strategy_comparison': 'results/strategy_comparison.json'
}

# Performance Thresholds (for highlighting good/bad performance)
PERFORMANCE_THRESHOLDS = {
    'excellent_reward': 0.15,
    'good_reward': 0.10,
    'poor_reward': 0.05,
    'excellent_fill_rate': 0.95,
    'good_fill_rate': 0.90,
    'poor_fill_rate': 0.80,
    'low_shortfall': 0.05,
    'medium_shortfall': 0.10,
    'high_shortfall': 0.15
}

# Colors for visualization
COLORS = {
    'rl_agent': '#1f77b4',
    'twap': '#ff7f0e', 
    'vwap': '#2ca02c',
    'market_order': '#d62728',
    'background': '#f0f2f6',
    'success': '#00c851',
    'warning': '#ffbb33',
    'danger': '#ff4444'
}

# Research Paper Information
RESEARCH_INFO = {
    'title': 'Optimal Trade Execution using Reinforcement Learning',
    'authors': 'Research Team',
    'abstract': 'Implementation of deep reinforcement learning for optimal trade execution using Actor-Critic methods with logistic-normal action distributions.',
    'keywords': ['Reinforcement Learning', 'Trade Execution', 'Market Microstructure', 'PyTorch'],
    'github_url': 'https://github.com/your-repo/rlte'
}
