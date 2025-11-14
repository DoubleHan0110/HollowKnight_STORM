"""
HollowKnight 环境专用训练脚本

注意：
1. HollowKnight 环境只能单个运行，不支持并行环境
2. 环境构建方式与 eval_HK.py 保持一致，确保训练和评估表现一致
3. 不使用 MaxLast2FrameSkipWrapper
"""

import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import shutil
import os
import sys

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.world_models import WorldModel
from HollowKnight_env.HKenv import HKEnv
import time

def build_hk_env(image_size, seed):
    """
    构建 HollowKnight 环境
    
    与 eval_HK.py 中的 build_hk_env 保持一致，确保训练和评估使用相同的环境配置
    """
    env = HKEnv()
    # Convert MultiBinary to Discrete
    env = env_wrapper.MultiBinaryToDiscreteWrapper(env)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    # 注意：不使用 MaxLast2FrameSkipWrapper，与 eval_HK.py 保持一致
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, 
                           batch_size, demonstration_batch_size, batch_length, logger):
    """训练世界模型一步"""
    obs, action, reward, termination = replay_buffer.sample(
        batch_size, demonstration_batch_size, batch_length
    )
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, 
                             agent: agents.ActorCriticAgent,
                             imagine_batch_size, 
                             imagine_demonstration_batch_size,
                             imagine_context_length, 
                             imagine_batch_length,
                             log_video, 
                             logger):
    """
    从 replay buffer 中采样上下文，然后使用世界模型和智能体生成想象数据
    """
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length
    )
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size + imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def train_hollowknight(max_steps, image_size,
                       replay_buffer: ReplayBuffer,
                       world_model: WorldModel, 
                       agent: agents.ActorCriticAgent,
                       train_dynamics_every_steps, 
                       train_agent_every_steps,
                       batch_size, 
                       demonstration_batch_size, 
                       batch_length,
                       imagine_batch_size, 
                       imagine_demonstration_batch_size,
                       imagine_context_length, 
                       imagine_batch_length,
                       save_every_steps, 
                       seed, 
                       logger,
                       run_name):
    """
    训练 HollowKnight 环境的主循环
    
    注意：使用单个环境，不支持并行
    """
    # 创建检查点目录
    os.makedirs(f"ckpt/{run_name}", exist_ok=True)

    # 构建单个环境（HollowKnight 不支持并行）
    env = build_hk_env(image_size, seed)
    print("Current env: " + colorama.Fore.YELLOW + "HollowKnight" + colorama.Style.RESET_ALL)

    # 重置环境并初始化变量
    sum_reward = 0.0
    current_obs, current_info = env.reset()
    # 添加 batch 维度以保持与模型输入格式一致
    current_obs = np.expand_dims(current_obs, axis=0)
    
    # 上下文缓冲区，用于存储最近的观测和动作
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    world_model_update_steps = 0
    agent_update_steps = 0
    last_save_steps = 0

    # 训练循环
    for total_steps in tqdm(range(max_steps), desc="Training"):
        # ========== 采样部分 ==========
        if replay_buffer.ready():
            # 如果 replay buffer 已准备好，使用模型生成动作
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    # 如果上下文为空，随机采样动作
                    action = env.action_space.sample()
                    action = np.array([action])  # 添加 batch 维度
                else:
                    # 使用 world model 和 agent 生成动作
                    # 编码上下文观测
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    # 准备上下文动作
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    # 计算特征
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                        context_latent, model_context_action
                    )
                    # 采样动作
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )

            # 更新上下文（添加 batch 维度）
            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W") / 255)
            context_action.append(action)
        else:
            # 如果 replay buffer 未准备好，随机采样动作
            action = env.action_space.sample()
            action = np.array([action])  # 添加 batch 维度

        # 执行动作（环境需要单个动作值，不是 batch）
        obs, reward, done, truncated, info = env.step(action[0])
        
        # 转换回 batch 格式以便存储到 replay buffer
        obs_batch = current_obs
        reward_batch = np.array([reward])
        done_batch = np.array([done])
        info_batch = {k: np.array([v]) if not isinstance(v, dict) else v for k, v in info.items()}

        next_obs_batch = np.expand_dims(obs, axis=0)
        
        # 存储到 replay buffer
        # 注意：replay_buffer.append 期望 batch 格式
        replay_buffer.append(obs_batch, action, reward_batch, 
                            np.logical_or(done_batch, info_batch["life_loss"]))
        
        
        # update 更新步数
        if replay_buffer.ready():
            world_model_update_steps += 1 / train_dynamics_every_steps
            agent_update_steps += 1 / train_agent_every_steps

        

        # 检查是否结束
        done_flag = np.logical_or(done_batch, np.array([truncated]))
        if done_flag.any():
            # 记录 episode 信息
            logger.log("sample/HollowKnight_reward", sum_reward)
            logger.log("sample/HollowKnight_episode_steps", current_info["episode_frame_number"])
            logger.log("replay_buffer/length", len(replay_buffer))

            if replay_buffer.ready():
                # print(f"开始训练world model 和 agent")
                # print(f"world_model_update_steps: {world_model_update_steps}, agent_update_steps: {agent_update_steps}")

                # ========== 训练世界模型部分 ==========
                while world_model_update_steps >= 1:
                    train_world_model_step(
                        replay_buffer=replay_buffer,
                        world_model=world_model,
                        batch_size=batch_size,
                        demonstration_batch_size=demonstration_batch_size,
                        batch_length=batch_length,
                        logger=logger
                    )
                    world_model_update_steps -= 1
                # ========== 训练世界模型部分结束 ==========

                # ========== 训练智能体部分 ==========
                has_logged_video = False
                while agent_update_steps >= 1:
                    # 决定是否记录视频（仅在保存检查点时记录）
                    if not has_logged_video and total_steps - last_save_steps >= save_every_steps:
                        log_video = True
                        has_logged_video = True
                    else:
                        log_video = False

                    # 生成想象数据
                    imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = \
                        world_model_imagine_data(
                            replay_buffer=replay_buffer,
                            world_model=world_model,
                            agent=agent,
                            imagine_batch_size=imagine_batch_size,
                            imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                            imagine_context_length=imagine_context_length,
                            imagine_batch_length=imagine_batch_length,
                            log_video=log_video,
                            logger=logger
                        )

                    # 更新智能体
                    agent.update(
                        latent=imagine_latent,
                        action=agent_action,
                        old_logprob=agent_logprob,
                        old_value=agent_value,
                        reward=imagine_reward,
                        termination=imagine_termination,
                        logger=logger
                    )
                    agent_update_steps -= 1
                # ========== 训练智能体部分结束 ==========
            # ========== 保存模型 ==========
            if total_steps - last_save_steps >= save_every_steps and total_steps > 0:
                print(colorama.Fore.GREEN + f"Saving model at step {total_steps}" + colorama.Style.RESET_ALL)
                torch.save(world_model.state_dict(), f"ckpt/{run_name}/world_model_{total_steps}.pth")
                torch.save(agent.state_dict(), f"ckpt/{run_name}/agent_{total_steps}.pth")
                last_save_steps = total_steps
            # 重置环境
            sum_reward = 0.0
            current_obs, current_info = env.reset()
            current_obs = np.expand_dims(current_obs, axis=0)
            # 清空上下文
            context_obs.clear()
            context_action.clear()
        else:
            # 更新当前状态
            sum_reward += reward
            current_obs = next_obs_batch
            current_info = info_batch

        
        
        # ========== 保存模型结束 ==========

    # 训练结束，关闭环境
    env.close()
    print(colorama.Fore.GREEN + "Training completed!" + colorama.Style.RESET_ALL)


def build_world_model(conf, action_dim):
    """构建世界模型"""
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads
    ).cuda()


def build_agent(conf, action_dim):
    """构建智能体"""
    return agents.ActorCriticAgent(
        feat_dim=32*32 + conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).cuda()


if __name__ == "__main__":
    # 忽略警告
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train on HollowKnight environment")
    parser.add_argument("-n", type=str, required=True, help="Run name")
    parser.add_argument("-seed", type=int, required=True, help="Random seed")
    parser.add_argument("-config_path", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("-trajectory_path", type=str, required=True, help="Path to demonstration trajectory (if using)")
    args = parser.parse_args()
    
    # 加载配置
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # 设置随机种子
    seed_np_torch(seed=args.seed)
    
    # 创建 TensorBoard logger
    logger = Logger(path=f"runs/{args.n}")
    
    # 复制配置文件到运行目录
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

    # 检查任务类型
    if conf.Task == "JointTrainAgent":
        # HollowKnight 环境的动作维度：MultiBinary(7) -> Discrete(128)
        action_dim = 128

        # 构建世界模型和智能体
        print(colorama.Fore.CYAN + "Building models..." + colorama.Style.RESET_ALL)
        world_model = build_world_model(conf, action_dim)
        agent = build_agent(conf, action_dim)
        print(colorama.Fore.CYAN + "Models built successfully!" + colorama.Style.RESET_ALL)

        # 构建 replay buffer
        # 注意：num_envs=1，因为 HollowKnight 不支持并行环境
        print(colorama.Fore.CYAN + "Building replay buffer..." + colorama.Style.RESET_ALL)
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,  
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU
        )
        print(colorama.Fore.CYAN + "Replay buffer built successfully!" + colorama.Style.RESET_ALL)

        # 判断是否加载演示轨迹
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + 
                  f"Loading demonstration trajectory from {args.trajectory_path}" + 
                  colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=args.trajectory_path)
            print(colorama.Fore.MAGENTA + "Demonstration trajectory loaded!" + colorama.Style.RESET_ALL)

        # 开始训练
        print(colorama.Fore.CYAN + "Starting training..." + colorama.Style.RESET_ALL)
        train_hollowknight(
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=args.seed,
            logger=logger,
            run_name=args.n
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")

