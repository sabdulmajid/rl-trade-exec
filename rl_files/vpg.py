'''
    - no value function 
    - no clipping 
    - only full trajectories, no value function bootstrapping
    - this is plain vanilla policy gradient
'''


# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os

import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet

from torch.utils.tensorboard import SummaryWriter

import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from simulation.market_gym import Market 

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    # env_id: str = "HalfCheetah-v4"
    env_id: str = "Market"
    """the id of the environment"""
    # total_timesteps: int = 1000000
    # total_timesteps: 200*70*100
    # total_timesteps: int = 1*100*256
    total_timesteps: int = 200*25*256
    """total timesteps of the experiments"""
    learning_rate: float = 1e-2
    """the learning rate of the optimizer"""
    # num_envs: int = 70
    # num_envs: int = 256
    num_envs: int = 256
    """the number of parallel game environments"""
    num_steps: int = 25
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 1.0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 10.0
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    # max_grad_norm: float = 0.5
    max_grad_norm: float = 100
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(config):
    def thunk():
        return Market(config)
    return thunk


def new_layer_init(layer):
    # note: this initialization worked worse than the default pytorch initialization
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    # torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    torch.nn.init.constant_(layer.weight, 0.0)
    return layer

class ActorNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(observation_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)
        self.tanh = nn.Tanh()
        self.sofplus = nn.Softplus()
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        # )
        # changed to reqires_grad=False
        # setting requires_grad=True yields better results: HOW IS THIS UPDATED ? 
        # I guess the parameters are not state dependent, but are updated as we go along
        # this must be the case otherwise entropy would not decreae gradually
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size), requires_grad=True)

    def forward(self, x):
        l1 = self.l1(x)
        l1 = self.tanh(l1)
        l2 = self.l2(l1)
        l2 = self.tanh(l2)
        l3 = self.l3(l2)
        concentrations = 1+self.sofplus(l3)
        return concentrations


# TODO implement an agent using Dirichlet policy 
class AgentDirichlet(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_network = ActorNetwork(np.array(envs.single_observation_space.shape).prod(), np.prod(envs.single_action_space.shape))
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)), requires_grad=True)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        concentrations = self.actor_network(x)
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        # not sure how action_std is updated by the optimizer 
        # action_std = torch.exp(action_logstd)
        probs = Dirichlet(concentrations)
        self.concentrations_params = concentrations
        if action is None:
            action = probs.sample()
        # entropy is summed for each component of the action
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # changed to reqires_grad=False
        # setting requires_grad=True yields better results: HOW IS THIS UPDATED ? 
        # i guess the parameters are not state dependent, but are updated as we go along
        # this must be the case otherwise entropy would not decreae gradually
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)), requires_grad=True)

    # def get_value(self, x):
    #     return self.critic(x)

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # not sure how action_std is updated by the optimizer 
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # entropy is summed for each component of the action
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


if __name__ == "__main__":
    ###
    volume = 60
    market_env = 'flow'
    ###
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(f'batch_size={args.batch_size}, minibatch_size={args.minibatch_size}, num_iterations={args.num_iterations}')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_{market_env}_{volume}_vanilla"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs_t200_std2/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    configs = [{'market_env': market_env, 'execution_agent': 'rl_agent', 'volume': volume, 'seed': args.seed+s} for s in range(args.num_envs)]
    env_fns = [make_env(config) for config in configs]
    envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)
    # envs = gym.vector.SyncVectorEnv(env_fns=env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    # print(agent)
    # agent = AgentDirichlet(envs).to(device)
    # print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game´
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device) 
    next_done = torch.zeros(args.num_envs).to(device)

    print('start training iterations')
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        returns = []
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            print(lrnow)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs)
                # values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        returns.append(info['cum_reward'])
                        # print(f"global_step={global_step}, episodic_return={info['cum_reward']}")
                        # if info and "episode" in info:
                        #     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        #     writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        #     writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        writer.add_scalar("charts/return", np.mean(returns), global_step)
        print(f'iteration={iteration}')
            

        # bootstrap value if not done
        # with torch.no_grad():
        #     next_value = agent.get_value(next_obs).reshape(1, -1)
        #     advantages = torch.zeros_like(rewards).to(device)
        #     lastgaelam = 0
        #     for t in reversed(range(args.num_steps)):
        #         if t == args.num_steps - 1:
        #             nextnonterminal = 1.0 - next_done
        #             nextvalues = next_value
        #         else:
        #             nextnonterminal = 1.0 - dones[t + 1]
        #             nextvalues = values[t + 1]
        #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        #     returns = advantages + values
        
        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        # no value bootstrapping
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            returns[args.num_steps-1] = rewards[args.num_steps-1] 
            for t in reversed(range(args.num_steps-1)):
                nextnonterminal = 1.0 - dones[t + 1]
                returns[t] = rewards[t] + returns[t+1] * nextnonterminal

            # slice to obtain only full episodes
            slices = [torch.where(dones[:, n]==1)[0][-1].item() for n in range(args.num_envs)]
            b_obs = torch.vstack([obs[:s, n].reshape((-1,) + envs.single_observation_space.shape) for s, n in zip(slices, range(args.num_envs))])
            b_logprobs = torch.hstack([logprobs[:s, n].reshape(-1,) for s, n in zip(slices, range(args.num_envs))])
            b_actions = torch.vstack([actions[:s, n].reshape((-1,) + envs.single_action_space.shape) for s, n in zip(slices, range(args.num_envs))])
            b_returns = torch.hstack([returns[:s, n].reshape(-1,) for s, n in zip(slices, range(args.num_envs))])
        

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                # mb_inds = b_inds[start:end]
                mb_inds = range(b_returns.shape[0])

                _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # mb_advantages = b_advantages[mb_inds]
                # just returns ! 
                # mb_advantages = returns[mb_inds]
                mb_advantages = torch.hstack([returns[:s, n].reshape(-1,) for s, n  in zip(slices, range(args.num_envs))])
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # pg_loss1 = -mb_advantages * ratio
                # pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # new
                pg_loss = torch.mean(-1*mb_advantages * newlogprob)
                # just returns 
                # pg_loss = torch.mean(-1*ret * newlogprob)

                # Value loss
                # newvalue = newvalue.view(-1)
                # if args.clip_vloss:
                #     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                #     v_clipped = b_values[mb_inds] + torch.clamp(
                #         newvalue - b_values[mb_inds],
                #         -args.clip_coef,
                #         args.clip_coef,
                #     )
                #     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                #     v_loss = 0.5 * v_loss_max.mean()
                # else:
                #     v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # not sure if it makes sense to take the mean here 
                # entropy should always be the same for each worker (or compment of the batch) (idx, action1, action2, action3)
                # entropy only depends on the current standard deviation 
                # instead of mean, could just take the first component of the entropy tensor
                entropy_loss = entropy.mean()

                loss = pg_loss 

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # print('concentrations')
        # print(agent.concentrations_params)

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs_t200_std2/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        # from cleanrl.cleanrl_utils.evals.ppo_eval import evaluate
        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=Agent,
        #     device=device,
        #     gamma=args.gamma,
        # )

        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()