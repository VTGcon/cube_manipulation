import sys

sys.path.append('../rrc_example_package`/')
sys.path.append('../../rrc_example_package')

import numpy as np
import env_wrapper as w
import torch


if __name__ == "__main__":
    device = torch.device('cpu')
    envs = w.create_vector_env(5)
    actor = w.Actor(envs).to(device)
    actor.load_state_dict(torch.load('model.zip', map_location=device))
    test_env = w.create_test_env(step_size=5)
    test_done = False
    test_state, test_cur_dist = w.reset_test_env(test_env)
    test_sum_reward = 0
    j = 0
    total_actions = []
    with torch.no_grad():
        while not test_done and j < 1200:
            test_action = actor(torch.FloatTensor(test_state).to(device))
            test_action = test_action.cpu().numpy().clip(
                test_env.action_space.low,
                test_env.action_space.high)
            total_actions.append(test_action.tolist())
            test_state, test_cur_reward, test_done, test_info, test_cur_dist = w.step_test_env(
                test_env,
                test_action,
                test_cur_dist,
                5
            )
            j += 1
            test_sum_reward += test_cur_reward