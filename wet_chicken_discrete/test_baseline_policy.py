import os
import sys

import numpy as np
import pandas as pd
from time import time

from shutil import copyfile

sys.path.append(r'C:\Users\phili\PycharmProjects\SPIBB')

import garnets
import spibb_utils
import spibb
import modelTransitions

from wet_chicken_discrete.dynamics import WetChicken
from wet_chicken_discrete.baseline_policy import WetChickenBaselinePolicy

if __name__ == '__main__':
    nb_iterations = 5
    seed = 1234
    np.random.seed(seed)
    log = True
    # nb_trajectories = 200
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
    max_nb_its = [500, 10 ** 3, 10 ** 3 * 5, 10 ** 4, 10 ** 4 * 5, 10 ** 5]
    # max_nb_its = [500, 10 ** 3]

    epsilon = 0.2
    order_epsilons = [np.inf]
    order_learning_rates = [3]
    results = []
    wet_chicken = WetChicken(length=length, width=width, max_turbulence=max_turbulence,
                             max_velocity=max_velocity)

    for iteration in range(nb_iterations):
        print(f'##### Starting with iteration {iteration + 1}/{nb_iterations} #####')
        for order_learning_rate in order_learning_rates:
            print(f'Starting with order learning rate {order_learning_rate} out of {order_learning_rates}.')
            for max_nb_it in max_nb_its:
                print(f'Starting with max_nb_it {max_nb_it} out of {max_nb_its}.')
                for order_epsilon in order_epsilons:
                    t_0 = time()

                    pi_baseline = WetChickenBaselinePolicy(env=wet_chicken, gamma=gamma,
                                                           method='state_count_dependent_variable',
                                                           order_epsilon=order_epsilon, learning_rate=learning_rate,
                                                           max_nb_it=max_nb_it, epsilon=epsilon,
                                                           order_learning_rate=order_learning_rate)
                    P = wet_chicken.get_transition_function()
                    R = wet_chicken.get_reward_function()
                    r_reshaped = spibb_utils.get_reward_model(P, R)


                    perf_baseline = spibb.policy_evaluation_exact(pi_baseline.pi, r_reshaped, P, gamma)[0][0]
                    print(f'Performance pi_baseline. {perf_baseline}')

                    pi_greedy = np.zeros((nb_states, nb_actions))
                    best_actions = np.argmax(pi_baseline.pi, axis=1)
                    for state in range(nb_states):
                        pi_greedy[state, best_actions[state]] = 1
                    pi_greedy_perf = spibb.policy_evaluation_exact(pi_greedy, r_reshaped, P, gamma)[0][0]
                    print(f'Performance pi_greedy. {pi_greedy_perf}')

                    pi_rand = np.ones((nb_states, nb_actions)) / nb_actions
                    pi_rand_perf = spibb.policy_evaluation_exact(pi_rand, r_reshaped, P, gamma)[0][0]
                    print(f'Performance pi_rand. {pi_rand_perf}')

                    rl = spibb.spibb(gamma, nb_states, nb_actions, pi_rand, None, P, r_reshaped, 'default')
                    rl.fit()
                    pi_star_perf = spibb.policy_evaluation_exact(rl.pi, r_reshaped, P, gamma)[0][0]
                    print(f'Performance pi_star. {pi_star_perf}')
#                    t_1 = time()
#                    time_calculating_baseline = t_1 - t_0
#
#                    t_0 = time()
#                    P = wet_chicken.get_transition_function()
#                    R = wet_chicken.get_reward_function()
#                    r_reshaped = spibb_utils.get_reward_model(P, R)
#                    t_1 = time()
#                    time_calculating_transition_fct = t_1 - t_0
#                    t_0 = time()
#                    perf_baseline = spibb.policy_evaluation_exact(pi_baseline.pi, r_reshaped, P, gamma)[0][0]
#                    t_1 = time()
#                    time_estimating_baseline = t_1 - t_0
#                    evaluation_method = 'Transition fct'
#                    results.append(
#                        [iteration, learning_rate, max_nb_it, epsilon, order_epsilon, order_learning_rate,
#                         evaluation_method, time_calculating_baseline, time_calculating_transition_fct,
#                         time_estimating_baseline, perf_baseline])
#
#    df = pd.DataFrame(results,
#                      columns=['iteration', 'learning_rate', 'max_nb_it', 'epsilon', 'order_epsilon',
#                               'order_learning_rate', 'evaluation_method', 'time_calculating_baseline',
#                               'time_calculating_transition_fct', 'time_estimating_baseline', 'perf_baseline'])
#    df.to_csv(r'results/testing_baseline_state_count_dependent_variable.csv')
