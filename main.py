# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('../rrc_example_package`/')
import env_wrapper
from experience_replay import ExpirienceReplay


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
                        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
                        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
                        help="the frequency of training policy (delayed)")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.seed}__{int(time.time())}"
    # import wandb
    # wandb.init(
    #     project=args.wandb_project_name,
    #     sync_tensorboard=True,
    #     config=vars(args),
    #     name='first run with reward=arctan(new_dist-old_dist)',
    #     monitor_gym=True,
    #     save_code=True,
    # )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = env_wrapper.create_vector_env()
    actor = env_wrapper.Actor(envs).to(device)
    qf1 = env_wrapper.QNetwork(envs).to(device)
    qf1_target = env_wrapper.QNetwork(envs).to(device)
    target_actor = env_wrapper.Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
    envs.single_observation_space.dtype = np.float32
    buffer = ExpirienceReplay(args.buffer_size)

    start_time = time.time()
    obs = env_wrapper.reset_env(envs)
    dist = np.sqrt((0.05 ** 2) * 2)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
        t1 = time.time()

        next_obs, rewards, dones, infos, new_dist = env_wrapper.step_env(envs, actions, dist)

        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        real_next_obs = next_obs.copy()

        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = env_wrapper.reset_env(envs)
        buffer.add((obs, actions, real_next_obs, rewards, dones))

        obs = next_obs
        dist = new_dist

        if global_step > args.learning_starts:
            t1 = time.time()
            data = buffer.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data[2].to(device))
                qf1_next_target = qf1_target(data[2].to(device), next_state_actions)
                next_q_value = data[3].to(device).flatten() + (1 - data[4].to(device).flatten()) * args.gamma * (
                    qf1_next_target).view(-1)
            t2 = time.time()
            qf1_a_values = qf1(data[0].to(device), data[1].to(device)).view(-1)
            qf1_loss = torch.nn.functional.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data[0].to(device), actor(data[0].to(device))).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            t3 = time.time()

            if global_step % 10000 == 0 or global_step == args.total_timesteps - 1:
                test_env = env_wrapper.create_test_env()
                test_done = False
                test_state = env_wrapper.reset_test_env(test_env)
                # TODO: remove hardcode
                test_cur_dist = np.sqrt((0.05 ** 2) * 2)
                test_sum_reward = 0
                j = 0
                with torch.no_grad():
                    while not test_done and j < 1000:
                        test_action = actor(torch.FloatTensor(test_state).to(device))
                        # print(test_action, test_state)
                        test_action = test_action.cpu().numpy().clip(test_env.action_space.low,
                                                                     test_env.action_space.high)
                        test_state, test_cur_reward, test_done, test_info, test_cur_dist = env_wrapper.step_test_env(
                            test_env,
                            test_action,
                            test_cur_dist)
                        j += 1
                        test_sum_reward += test_cur_reward

                writer.add_scalar("reward/step", test_sum_reward, global_step)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("first/SPS", (t2 - t1), global_step)
                writer.add_scalar("second/SPS", (t3 - t2), global_step)
                writer.add_scalar("charts/SPS", (global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
