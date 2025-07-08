# import os 
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from typing import Callable, List 
import gymnasium as gym
import torch
# from ppo_continuous_action import Agent, AgentDirichlet
from actor_critic import AgentLogisticNormal, DirichletAgent
# from ppo_modified import Agent
import numpy as np
import time 

def evaluate(
    env_fns: List[Callable],
    model_path: str,
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):      
    # env_fns = env_fns[:10]
    envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)
    print('environment is created')
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            # actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
            next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            # next_obs, _, _, _, infos = envs.step(actions.numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episodic_returns.append(info['cum_reward'])
                    # if "episode" not in info:
                    #     continue
                    # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    # episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

if __name__=="__main__":
    from simulation.market_gym import Market     
    # set up 
    env = 'flow'
    volume = 60
    n_cpus = 1
    #
    configs = [{'market_env': env, 'execution_agent': 'rl_agent', 'volume': volume, 'seed': 100+s} for s in range(n_cpus)]
    env_fns = [lambda: Market(config) for config in configs]
    # model_path = "runs/Market__ppo_continuous_action__0__1725462757_gaussian_20lots_more_features/ppo_continuous_action.cleanrl_model"
    # model_path = "runs/Market__ppo_continuous_action__0__1725470471_20lots_std3/ppo_continuous_action.cleanrl_model"

    # model_path = 'runs/Market__ppo_continuous_action__0__1725548983_20lots_std3/ppo_continuous_action.cleanrl_model'
    # 40 lots noise 
    # model_path = 'runs/Market__ppo_continuous_action__0__1725552339_20lots_std3/ppo_continuous_action.cleanrl_model'
    # 20 lots flow 
    # model_path =  'runs/Market__ppo_continuous_action__0__1725555789_flow_20/ppo_continuous_action.cleanrl_model'
    # 40 lots flow 
    # model_path = 'runs/Market__ppo_continuous_action__0__1725559747_flow_40/ppo_continuous_action.cleanrl_model'
    # without clipping 
    # model_path = 'runs/Market__ppo_modified__0__1725902028_flow_40_noclip/ppo_modified.cleanrl_model'
    # model_path = 'runs/Market__ppo_modified__0__1725914125_flow_40_noclip/ppo_modified.cleanrl_model'
    # model path with queues 
    # model_path = 'runs/Market__ppo_continuous_action__0__1726753861_flow_40_with_queues/ppo_continuous_action.cleanrl_model'
    # model_path = 'runs/Market__ppo_continuous_action__0__1726767202_strategic_40/ppo_continuous_action.cleanrl_model'
    # model_path = 'runs_std4/Market__ppo_continuous_action__0__1726841027_flow_60/ppo_continuous_action.cleanrl_model'
    # model_path = 'runs_std4/Market__ppo_continuous_action__0__1727359648_flow_60/ppo_continuous_action.cleanrl_model'
    # model_path = 'runs_t200_std2/Market__ppo_continuous_action__0__1727729743_flow_60/ppo_continuous_action.cleanrl_model'
    # model_path = 'runs_t200_std2/Market__actor_critic__0__1727807989_flow_60/actor_critic.cleanrl_model'
    # model_path = 'runs_t150_std2/Market__actor_critic__0__1727814771_flow_60/actor_critic.cleanrl_model'
    # model_path = 'runs_t150_std2/Market__actor_critic__0__1727855647_flow_20/actor_critic.cleanrl_model'
    # model_path = 'runs_t150_std2/Market__actor_critic__0__1727867325_flow_20/actor_critic.cleanrl_model'
    # model_path = 'runs_t150_std2/Market__actor_critic__0__1727873682_noise_20/actor_critic.cleanrl_model'
    # flow 60 dirichlet 

    # model_path = 'runs_t150_std2/Market__actor_critic__0__1728480261_flow_60_dirichlet/actor_critic.cleanrl_model'

    model_path = 'runs_t150/Market__actor_critic__0__1729712129_flow_60_logistic_normal_learnable_variance/actor_critic.cleanrl_model'

    t = time.time()
    returns = evaluate(
        model_path=model_path,
        env_fns=env_fns,
        eval_episodes=4000,
        Model=AgentLogisticNormal,
        # device="cpu",
    )
    
    print(np.mean(returns))
    print(f"elapsed time: {time.time()-t}")

    file_name = f'raw_rewards/std3_t200_rewards_{env}_{volume}_rl_agent_dirichlet.npz'
    print(f'save to {file_name}')
    np.savez(file_name, rewards=returns)


