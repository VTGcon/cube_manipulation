import gym
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from rrc_example_package.cube_trajectory_env import SimCubeTrajectoryEnv, ActionType


def normalize_space(space):
    robot_pos = space['robot_observation']['position']
    robot_vel = space['robot_observation']['velocity']
    robot_torque = space['robot_observation']['torque']
    obj_pos = space['object_observation']['position']
    obj_orientation = space['object_observation']['orientation']
    achieved_goal = space['achieved_goal']
    goal_pos = space['desired_goal']
    if not isinstance(space, gym.spaces.Dict):
        ans = []
        for i in range(len(robot_pos)):
            robot_pos_new = np.copy(robot_pos[i]) - np.tile(goal_pos, 3)
            obj_pos_new = np.copy(obj_pos[i]) - goal_pos
            achieved_goal_new = np.copy(achieved_goal[i]) - goal_pos
            obs = np.array(robot_vel[i])
            obs = np.append(obs, robot_torque[i])
            obs = np.append(obs, robot_pos_new)
            obs = np.append(obs, obj_pos_new)
            obs = np.append(obs, obj_orientation[i])
            obs = np.append(obs, achieved_goal_new)
            ans.append(obs)
        return np.array(ans)
    robot_pos_low = np.copy(robot_pos.low) - np.tile(goal_pos.high, 3)
    obj_pos_low = np.copy(obj_pos.low) - goal_pos.high
    achieved_goal_low = np.copy(achieved_goal.low) - goal_pos.high
    robot_pos_high = np.copy(robot_pos.high) - np.tile(goal_pos.low, 3)
    obj_pos_high = np.copy(obj_pos.high) - goal_pos.low
    achieved_goal_high = np.copy(achieved_goal.high) - goal_pos.low

    obs_low = np.array([robot_vel.low])
    obs_low = np.append(obs_low, robot_torque.low)
    obs_low = np.append(obs_low, robot_pos_low)
    obs_low = np.append(obs_low, obj_pos_low)
    obs_low = np.append(obs_low, obj_orientation.low)
    obs_low = np.append(obs_low, achieved_goal_low)

    obs_high = np.array([robot_vel.high])
    obs_high = np.append(obs_high, robot_torque.high)
    obs_high = np.append(obs_high, robot_pos_high)
    obs_high = np.append(obs_high, obj_pos_high)
    obs_high = np.append(obs_high, obj_orientation.high)
    obs_high = np.append(obs_high, achieved_goal_high)

    return gym.spaces.Box(obs_low, obs_high)


def normalize_test_space(space):
    robot_pos = np.copy(space['robot_observation']['position'])
    robot_vel = np.copy(space['robot_observation']['velocity'])
    robot_torque = np.copy(space['robot_observation']['torque'])
    obj_pos = np.copy(space['object_observation']['position'])
    obj_orientation = np.copy(space['object_observation']['orientation'])
    achieved_goal = np.copy(space['achieved_goal'])
    goal_pos = np.copy(space['desired_goal'])
    robot_pos -= np.tile(goal_pos, 3)
    obj_pos -= goal_pos
    achieved_goal -= goal_pos

    obs = np.array(robot_vel)
    obs = np.append(obs, robot_torque)
    obs = np.append(obs, robot_pos)
    obs = np.append(obs, obj_pos)
    obs = np.append(obs, obj_orientation)
    obs = np.append(obs, achieved_goal)

    return obs


def make_env(step_size, goal_trajectory=None):

    def create_env():
        env = SimCubeTrajectoryEnv(goal_trajectory=goal_trajectory, action_type=ActionType.TORQUE, step_size=step_size)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return create_env


def create_vector_env(step_size, goal_trajectory=None):
    return gym.vector.SyncVectorEnv([make_env(step_size, goal_trajectory)])


def create_test_env(step_size, goal_trajectory=None):
    env = SimCubeTrajectoryEnv(goal_trajectory=goal_trajectory, action_type=ActionType.TORQUE, visualization=False,
                               step_size=step_size)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def reset_env(env):
    obs = env.reset()
    return normalize_space(obs), np.sqrt(np.sum((obs['object_observation']['position'] - obs['desired_goal']) ** 2))


def reset_test_env(env):
    obs = env.reset()
    return normalize_test_space(obs), np.sqrt(
        np.sum((obs['object_observation']['position'] - obs['desired_goal']) ** 2))


def step_env(env, action, prev_dist, step_size):
    next_obs, rewards, dones, infos = env.step(action)
    new_dist = np.sqrt(np.sum((next_obs['object_observation']['position'] - next_obs['desired_goal']) ** 2))
    reward = np.array([1000 * (prev_dist - new_dist) - 0.001])
    if new_dist < 0.065 / 10:
        dones = [True]
    if infos[0]['time_index'] > 1200 * step_size:
        dones = [True]
    return normalize_space(next_obs), reward, dones, infos, new_dist


def step_test_env(env, action, prev_dist, step_size):
    next_obs, rewards, dones, infos = env.step(action)
    new_dist = np.sqrt(np.sum((next_obs['object_observation']['position'] - next_obs['desired_goal']) ** 2))
    reward = 1000 * (prev_dist - new_dist) - 0.001
    if new_dist < 0.065 / 10:
        dones = True
    if infos['time_index'] > 1200 * step_size:
        dones = True
    # print(reward, next_obs)
    return normalize_test_space(next_obs), reward, dones, infos, new_dist


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        sop = normalize_space(env.single_observation_space)
        self.fc1 = nn.Linear(np.array(sop.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        sos = normalize_space(env.single_observation_space)
        self.fc1 = nn.Linear(np.array(sos.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space[0].high - env.action_space[0].low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space[0].high + env.action_space[0].low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
