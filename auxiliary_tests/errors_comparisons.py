# File to explore the difference between the error function relying on Hoeffding's bound and the one relying on the
# bound of Maurer and Pontil.
import os
import sys
import configparser

import numpy as np


directory = os.path.dirname(os.path.dirname(os.path.expanduser(__file__)))
sys.path.append(directory)
path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'paths.ini'))
spibb_path = path_config['PATHS']['spibb_path']
sys.path.append(spibb_path)

from wet_chicken_discrete.dynamics import WetChicken
from wet_chicken_discrete.baseline_policy import WetChickenBaselinePolicy

from batch_rl_algorithms.soft_spibb import ApproxSoftSPIBB
import spibb_utils


if __name__ == '__main__':
    nb_iterations = 50
    seed = 1602421836
    seed = 1
    np.random.seed(seed)
    log = True
    # ratio = 0.9
    epsilon = 0.1
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

    epsilon_baseline = 0.1
    order_epsilon = np.inf
    order_learning_rate = 3
    episodic = False

    results = []

    wet_chicken = WetChicken(length=length, width=width, max_turbulence=max_turbulence,
                             max_velocity=max_velocity)

    pi_baseline = WetChickenBaselinePolicy(env=wet_chicken, gamma=gamma, method='heuristic',
                                           order_epsilon=order_epsilon, learning_rate=learning_rate,
                                           max_nb_it=max_nb_it, epsilon=epsilon_baseline,
                                           order_learning_rate=order_learning_rate)
    pi_b = pi_baseline.pi
    P = wet_chicken.get_transition_function()
    R = wet_chicken.get_reward_function()
    r_reshaped = spibb_utils.get_reward_model(P, R)

    length_trajectory = 10000
    trajectory = spibb_utils.generate_batch_wet_chicken(length_trajectory, wet_chicken, pi_b)


    approx_soft_spibb = ApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                        data=trajectory, R=R, delta=delta, epsilon=epsilon,
                                        error_kind='hoeffding', episodic=episodic, checks=False)
    e_hoeffding = np.nan_to_num(approx_soft_spibb.errors, nan=0, posinf=0)

    approx_soft_spibb = ApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                        data=trajectory, R=R, delta=delta, epsilon=epsilon,
                                        error_kind='mpeb', episodic=episodic, checks=False, g_max=40)

    e_mpeb = np.nan_to_num(approx_soft_spibb.errors, nan=0, posinf=0)

    print(f'L1 distance (interpreted as long vector instead of matrix) : {np.sum(np.abs(e_hoeffding - e_mpeb))}')

    # count_state_action = approx_soft_spibb.count_state_action

    print(f'Hi')
