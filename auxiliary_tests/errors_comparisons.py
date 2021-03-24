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

from batch_rl_algorithms.basic_rl import Basic_rl
from batch_rl_algorithms.pi_star import PiStar
from batch_rl_algorithms.spibb import SPIBB, Lower_SPIBB
from batch_rl_algorithms.r_min import RMin
from batch_rl_algorithms.soft_spibb import ApproxSoftSPIBB, ExactSoftSPIBB, LowerApproxSoftSPIBB, AdvApproxSoftSPIBB
from batch_rl_algorithms.duipi import DUIPI
from batch_rl_algorithms.ramdp import RaMDP
from batch_rl_algorithms.mbie import MBIE

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
                                        error_kind='mpeb', episodic=episodic, checks=False)

    e_mpeb = np.nan_to_num(approx_soft_spibb.errors, nan=0, posinf=0)

    approx_soft_spibb = ApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                        data=trajectory, R=R, delta=delta, epsilon=epsilon,
                                        error_kind='min', episodic=episodic, checks=False)

    e_min = np.nan_to_num(approx_soft_spibb.errors, nan=0, posinf=0)

    print(f'L1 distance (interpreted as long vector instead of matrix) : {np.sum(np.abs(e_hoeffding - e_min))}')

    count_state_action = approx_soft_spibb.count_state_action

    print(f'Hi')
