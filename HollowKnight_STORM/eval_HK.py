import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import os
import sys
import glob

from utils import seed_np_torch, load_config
import env_wrapper
import agents
from sub_models.world_models import WorldModel
from HollowKnight_env.HKenv import HKEnv
import train


def build_hk_env(image_size, seed):
    """构建 HollowKnight 环境，与 train.py 保持一致"""
    env = HKEnv()
    # Convert MultiBinary to Discrete
    env = env_wrapper.MultiBinaryToDiscreteWrapper(env)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    # 注意：不使用 MaxLast2FrameSkipWrapper，与 train.py 保持一致
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def eval_episodes(num_episode, image_size, world_model: WorldModel, 
                  agent: agents.ActorCriticAgent, seed=0):
    """评估 HollowKnight 环境"""
    world_model.eval()
    agent.eval()
    
    env = build_hk_env(image_size, seed)
    print("Current env: " + colorama.Fore.YELLOW + "HollowKnight" + colorama.Style.RESET_ALL)
    
    sum_reward = 0.0
    current_obs, current_info = env.reset()
    current_obs = np.expand_dims(current_obs, axis=0)  # Add batch dimension
    
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    
    final_rewards = []
    episode_count = 0
    
    while episode_count < num_episode:
        with torch.no_grad():
            if len(context_action) == 0:
                # 第一个动作随机采样
                action = env.action_space.sample()
                action = np.array([action])
            else:
                # 使用 world model 和 agent 生成动作
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                    context_latent, model_context_action
                )
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=True  # 评估时使用贪婪策略
                )
        
        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_action.append(action)
        
        obs, reward, done, truncated, info = env.step(action[0])
        
        # Convert to batch format
        obs = np.expand_dims(obs, axis=0)
        reward = np.array([reward])
        done = np.array([done])
        truncated = np.array([truncated])
        info = {k: np.array([v]) if not isinstance(v, dict) else v for k, v in info.items()}

        sum_reward += reward[0]
        
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            final_rewards.append(sum_reward)
            episode_count += 1
            print(f"Episode {episode_count}/{num_episode}: reward = {sum_reward:.2f}")
            
            if episode_count < num_episode:
                # Reset for next episode
                sum_reward = 0.0
                current_obs, current_info = env.reset()
                current_obs = np.expand_dims(current_obs, axis=0)
                context_obs.clear()
                context_action.clear()
            else:
                break
        
        # Update state
        current_obs = obs
        current_info = info
    
    env.close()
    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    print("=" * 50)
    print("Evaluation Results:")
    print(f"  Episodes: {num_episode}")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Std Reward: {std_reward:.2f}")
    print(f"  Min Reward: {np.min(final_rewards):.2f}")
    print(f"  Max Reward: {np.max(final_rewards):.2f}")
    print("=" * 50)
    
    return mean_reward


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("-run_name", type=str, required=True, help="Run name (checkpoint directory name)")
    parser.add_argument("-num_episode", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("-step", type=int, default=None, help="Specific checkpoint step (default: latest)")
    args = parser.parse_args()
    
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + f"Config: {args.config_path}" + colorama.Style.RESET_ALL)
    print(colorama.Fore.RED + f"Run: {args.run_name}" + colorama.Style.RESET_ALL)
    
    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)
    
    # Build models
    action_dim = 128  # HollowKnight: MultiBinary(7) -> Discrete(128)
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    
    # Load checkpoint
    root_path = f"ckpt/{args.run_name}"
    if args.step is not None:
        steps = [args.step]
    else:
        # Load latest checkpoint
        pathes = glob.glob(f"{root_path}/world_model_*.pth")
        if not pathes:
            raise FileNotFoundError(f"No checkpoints found in {root_path}")
        steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
        steps.sort()
        steps = steps[-1:]  # Latest checkpoint
    
    print(f"Evaluating checkpoint at step: {steps[0]}")
    print(f"Model files: world_model_{steps[0]}.pth, agent_{steps[0]}.pth")
    
    # Evaluate
    results = []
    for step in steps:
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth", map_location='cuda'))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth", map_location='cuda'))
        
        episode_avg_return = eval_episodes(
            num_episode=args.num_episode,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent,
            seed=conf.BasicSettings.Seed
        )
        results.append([step, episode_avg_return])
    
    # Save results
    os.makedirs("eval_result", exist_ok=True)
    result_file = f"eval_result/{args.run_name}_step{steps[0]}.csv"
    with open(result_file, "w") as fout:
        fout.write("step,episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")
    
    print(f"\nResults saved to: {result_file}")

