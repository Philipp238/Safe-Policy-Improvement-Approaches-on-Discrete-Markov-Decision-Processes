import os
import sys

import numpy as np
import pandas as pd
import time

from shutil import copyfile

sys.path.append(r'C:\Users\phili\PycharmProjects\SPIBB')

import garnets
import spibb_utils
import spibb
import modelTransitions

from wet_chicken_discrete.dynamics import WetChicken
from wet_chicken_discrete.baseline_policy import WetChickenBaselinePolicy
from wet_chicken_discrete.baseline_policy import continuous_heuristic_policy

from batch_rl_algorithms.basic_rl import Basic_rl
from batch_rl_algorithms.pi_star import PiStar
from batch_rl_algorithms.spibb import SPIBB
from batch_rl_algorithms.spibb import Lower_SPIBB
from batch_rl_algorithms.r_min import RMin
from batch_rl_algorithms.soft_spibb import ApproxSoftSPIBB
from batch_rl_algorithms.soft_spibb import ExactSoftSPIBB
from batch_rl_algorithms.soft_spibb import LowerApproxSoftSPIBB
from batch_rl_algorithms.soft_spibb import AdvApproxSoftSPIBB
from batch_rl_algorithms.duipi import DUIPI
from batch_rl_algorithms.ramdp import RaMDP
from batch_rl_algorithms.mbie import MBIE

if __name__ == '__main__':
    nb_iterations = 50
    seed = 1469231697
    np.random.seed(seed)
    log = True
    epsilon = 4
    # ratio = 0.9
    delta = 1

    gamma = 0.95
    length = 5
    width = 5
    max_turbulence = 3.5
    max_velocity = 3

    nb_states = length * width
    nb_actions = 5

    learning_rate = 0.5
    max_nb_it = 10 ** 5

    epsilon = 0.1
    order_epsilon = np.inf
    order_learning_rate = 3
    episodic = False

    results = []

    wet_chicken = WetChicken(length=length, width=width, max_turbulence=max_turbulence,
                             max_velocity=max_velocity)

    pi_baseline = WetChickenBaselinePolicy(env=wet_chicken, gamma=gamma, method='heuristic',
                                           order_epsilon=order_epsilon, learning_rate=learning_rate,
                                           max_nb_it=max_nb_it, epsilon=epsilon,
                                           order_learning_rate=order_learning_rate)
    pi_b = pi_baseline.pi

    P = wet_chicken.get_transition_function()
    R = wet_chicken.get_reward_function()
    r_reshaped = spibb_utils.get_reward_model(P, R)

    # perf_baseline = spibb.policy_evaluation_exact(pi_b, r_reshaped, P, gamma)[0][0]
    # print(f'Performance pi_baseline. {perf_baseline}')
    #
    # pi_greedy = np.zeros((nb_states, nb_actions))
    # best_actions = np.argmax(pi_baseline.pi, axis=1)
    # for state in range(nb_states):
    #     pi_greedy[state, best_actions[state]] = 1
    # pi_greedy_perf = spibb.policy_evaluation_exact(pi_greedy, r_reshaped, P, gamma)[0][0]
    # print(f'Performance pi_greedy. {pi_greedy_perf}')
    #
    # pi_rand = np.ones((nb_states, nb_actions)) / nb_actions
    # pi_rand_perf = spibb.policy_evaluation_exact(pi_rand, r_reshaped, P, gamma)[0][0]
    # print(f'Performance pi_rand. {pi_rand_perf}')

    # rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, None, P, r_reshaped, 'default')
    # rl.fit()
    # pi_star_perf = spibb.policy_evaluation_exact(rl.pi, r_reshaped, P, gamma)[0][0]
    # print(f'Performance pi_star. {pi_star_perf}')

    # pi_star_epsilon_greedy = epsilon * np.ones((nb_states, nb_actions)) / 5
    # for s in range(nb_states):
    #     pi_star_epsilon_greedy[s, np.argmax(rl.pi[s, :])] += 1 - epsilon
    # pi_star_epsilon_greedy_perf = spibb.policy_evaluation_exact(pi_star_epsilon_greedy, r_reshaped, P, gamma)[0][0]
    # print(f'Performance pi_star_epsilon_greedy. {pi_star_epsilon_greedy_perf}')
    #
    # pi_1 = np.zeros((25, 5))
    # pi_1[:, 0] = 1
    # pi_1_perf = spibb.policy_evaluation_exact(pi_1, r_reshaped, P, gamma)[0][0]
    # print(f'Performance pi_1. {pi_1_perf}')

    length_trajectory = 1000000

    trajectory = spibb_utils.generate_batch_wet_chicken(length_trajectory, wet_chicken, pi_b)

    approx_soft_spibb_hoeffding = ApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                                  data=trajectory, R=R, delta=delta, epsilon=epsilon,
                                                  error_kind='hoeffding', episodic=episodic, checks=False)
    errors_hoeffding = np.nan_to_num(approx_soft_spibb_hoeffding.errors, nan=0)
    approx_soft_spibb_e_min = ApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                              data=trajectory, R=R, delta=delta, epsilon=epsilon,
                                              error_kind='min', episodic=episodic, checks=False)
    errors_e_min = np.nan_to_num(approx_soft_spibb_e_min.errors, nan=0)

    print(f'Distance: {np.linalg.norm(errors_hoeffding - errors_e_min)}')

    model = modelTransitions.ModelTransitions(trajectory, nb_states, nb_actions)
    reward_model = spibb_utils.get_reward_model(model.transitions, R)

    count_state_action = 0.00001 * np.ones((nb_states, nb_actions))
    kappa = 0.003
    for [action, state, next_state, reward] in trajectory:
        count_state_action[state, action] += 1

    ramdp_reward_model = reward_model - kappa / np.sqrt(count_state_action)
    ramdp = spibb.spibb(gamma, nb_states, nb_actions, pi_b, None, model.transitions,
                        ramdp_reward_model,
                        'default')
    ramdp.fit()
    # Evaluates the Reward-adjusted MDP RL policy performance
    perf_RaMDP = spibb.policy_evaluation_exact(ramdp.pi, r_reshaped, P, gamma)[0][0]
    print(f'perf_RaMDP_old: {perf_RaMDP}')
    # Compute RaMDP with batch_rl_algorithms.ramdp
    ramdp_new = RaMDP(pi_b=pi_b, gamma=gamma, nb_states=nb_states,
                      nb_actions=nb_actions, data=trajectory, episodic=episodic,
                      R=R, kappa=kappa)
    ramdp_new.fit()
    ramdp_new_perf = spibb.policy_evaluation_exact(ramdp_new.pi, r_reshaped, P, gamma)[0][0]
    print(f'RaMDP_new: {ramdp_new_perf}')

    print('hi')

