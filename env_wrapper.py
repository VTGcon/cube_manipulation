import gym
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from rrc_example_package.cube_trajectory_env import SimCubeTrajectoryEnv, ActionType


def normalize_space(space):
    robot_pos = space['robot_observation']['position']
    # print(robot_pos)
    robot_vel = space['robot_observation']['velocity']
    robot_torque = space['robot_observation']['torque']
    obj_pos = space['object_observation']['position']
    obj_orientation = space['object_observation']['orientation']
    desired_goal = space['desired_goal']
    achieved_goal = space['achieved_goal']
    if not isinstance(space, gym.spaces.Dict):
        ans = []
        for i in range(len(robot_pos)):
            robot_pos[i] -= np.tile(desired_goal[i], 3)
            obj_pos[i] -= desired_goal[i]
            achieved_goal[i] -= desired_goal[i]
            ans.append(np.append(robot_vel[i], np.append(robot_torque[i], np.append(robot_pos[i],
                                                                                    np.append(obj_pos[i],
                                                                                              np.append(
                                                                                                  obj_orientation[
                                                                                                      i],
                                                                                                  np.append(
                                                                                                      desired_goal[
                                                                                                          i],
                                                                                                      achieved_goal[
                                                                                                          i])))))))
        return np.array(ans)
    robot_pos.low -= np.tile(desired_goal.low, 3)
    obj_pos.low -= desired_goal.low
    achieved_goal.low -= desired_goal.low
    robot_pos.high -= np.tile(desired_goal.high, 3)
    obj_pos.high -= desired_goal.high
    achieved_goal.high -= desired_goal.high
    return gym.spaces.Box(np.append(robot_vel.low, np.append(robot_torque.low, np.append(robot_pos.low,
                                                                                         np.append(obj_pos.low,
                                                                                                   np.append(
                                                                                                       obj_orientation.low,
                                                                                                       np.append(
                                                                                                           desired_goal.low,
                                                                                                           achieved_goal.low)))))),
                          np.append(robot_vel.high, np.append(robot_torque.high, np.append(robot_pos.high,
                                                                                           np.append(obj_pos.high,
                                                                                                     np.append(
                                                                                                         obj_orientation.high,
                                                                                                         np.append(
                                                                                                             desired_goal.high,
                                                                                                             achieved_goal.high)))))),
                          )


def make_env(goal_trajectory=None):
    if goal_trajectory is None:
        goal_trajectory = [(0, [0.05, 0.05, 0.0325])]

    def create_env():
        env = SimCubeTrajectoryEnv(goal_trajectory=goal_trajectory, action_type=ActionType.POSITION)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return create_env


def create_vector_env(goal_trajectory=None):
    if goal_trajectory is None:
        goal_trajectory = [(0, [0.05, 0.05, 0.0325])]
    return gym.vector.SyncVectorEnv([make_env(goal_trajectory)])


def reset_env(env):
    return normalize_space(env.reset())


def step_env(env, action, prev_dist):
    next_obs, rewards, dones, infos = env.step(action)
    new_dist = np.sum((next_obs['object_observation']['position'] - next_obs['desired_goal']) ** 2)
    return normalize_space(next_obs), np.array([np.arctan((new_dist - prev_dist))]), dones, infos, new_dist


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
        # action rescaling
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
