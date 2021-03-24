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

from sklearn.preprocessing import normalize
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


def prepare_experiments(delta, nb_trajectories, epsilon, ratio, gamma, nb_states, nb_actions,
                        nb_next_state_transition, env_type, self_transitions):
    garnet = garnets.Garnets(nb_states, nb_actions, nb_next_state_transition,
                             env_type=env_type, self_transitions=self_transitions)

    softmax_target_perf_ratio = (ratio + 1) / 2
    baseline_target_perf_ratio = ratio
    pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf = \
        garnet.generate_baseline_policy(gamma,
                                        softmax_target_perf_ratio=softmax_target_perf_ratio,
                                        baseline_target_perf_ratio=baseline_target_perf_ratio, log=False)

    reward_current = garnet.compute_reward()
    current_proba = garnet.transition_function

    if env_type == 2:  # easter
        # Randomly pick a second terminal state and update model parameters
        potential_final_states = [s for s in range(nb_states) if s != garnet.final_state and s != 0]
        easter_egg = np.random.choice(potential_final_states)
        # Or pick the one with the least transitions
        # current_proba_sum = current_proba.reshape(-1, current_proba.shape[-1]).sum(axis=0)
        # mask_easter = np.ma.array(current_proba_sum, mask=False)
        # mask_easter.mask[garnet.final_state] = True
        # easter_egg = np.argmin(mask_easter)
        assert (garnet.final_state != easter_egg)
        reward_current[:, easter_egg] = 1
        current_proba[easter_egg, :, :] = 0
        r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)
        # Compute optimal policy in this new environment
        ### PHILIPP: First mask_0 should be pi_b?
        true_rl = spibb.spibb(gamma, nb_states, nb_actions, mask_0, mask_0, current_proba, r_reshaped,
                              'default')
        true_rl.fit()
        pi_star_perf = spibb.policy_evaluation_exact(true_rl.pi, r_reshaped, current_proba, gamma)[0][0]
        pi_b_perf = spibb.policy_evaluation_exact(pi_b, r_reshaped, current_proba, gamma)[0][0]
        if log:
            print("Optimal perf in easter egg environment:\t\t\t" + str(pi_star_perf))
            print("Baseline perf in easter egg environment:\t\t\t" + str(pi_b_perf))
    elif env_type == 3:
        # Randomly pick a second terminal state and update model parameters
        potential_final_states = [s for s in range(nb_states) if s != garnet.final_state and s != 0]
        easter_egg = np.random.choice(potential_final_states)
        # Or pick the one with the least transitions
        # current_proba_sum = current_proba.reshape(-1, current_proba.shape[-1]).sum(axis=0)
        # mask_easter = np.ma.array(current_proba_sum, mask=False)
        # mask_easter.mask[garnet.final_state] = True
        # easter_egg = np.argmin(mask_easter)
        assert (garnet.final_state != easter_egg)
        reward_current[:, easter_egg] = -1
        current_proba[easter_egg, :, :] = 0
        r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)
        # Compute optimal policy in this new environment
        ### PHILIPP: First mask_0 should be pi_b?
        true_rl = spibb.spibb(gamma, nb_states, nb_actions, mask_0, mask_0, current_proba, r_reshaped,
                              'default')
        true_rl.fit()
        pi_star_perf = spibb.policy_evaluation_exact(true_rl.pi, r_reshaped, current_proba, gamma)[0][0]
        pi_b_perf = spibb.policy_evaluation_exact(pi_b, r_reshaped, current_proba, gamma)[0][0]
        if log:
            print("Optimal perf in bad easter egg environment:\t\t\t" + str(pi_star_perf))
            print("Baseline perf in bad easter egg environment:\t\t\t" + str(pi_b_perf))
    else:
        easter_egg = None
        r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)

    # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
    trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b, easter_egg=easter_egg)

    t_0 = time.time()
    model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
    t_1 = time.time()
    print(f'Time to calc the model {t_1 - t_0}')
    reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)
    q_pi_b_est = spibb_utils.compute_q_pib_est_episodic(gamma=gamma, nb_actions=nb_actions, nb_states=nb_states,
                                                        batch=trajectories)
    errors = spibb.compute_errors(nb_states, nb_actions, delta, batch_traj)
    return pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf, reward_current, current_proba, r_reshaped, \
           trajectories, batch_traj, model, reward_model, q_pi_b_est, errors


if __name__ == '__main__':
    episodic = True
    seed = 123456
    np.random.seed(seed)
    log = True
    nb_trajectories = 2000
    epsilon = 4
    ratio = 0.9
    delta = 1
    bayesian = True

    gamma = 0.95
    nb_states = 50
    nb_actions = 4
    nb_next_state_transition = 4
    env_type = 2  # 1 for one terminal state, 2 for two terminal states

    self_transitions = 0

    mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
    mask_0 = ~mask_0
    rand_pi = np.ones((nb_states, nb_actions)) / nb_actions

    pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf, reward_current, current_proba, r_reshaped, trajectories, \
    batch_traj, model, reward_model, q_pi_b_est, errors = prepare_experiments(delta, nb_trajectories, epsilon, ratio,
                                                                              gamma,
                                                                              nb_states, nb_actions,
                                                                              nb_next_state_transition,
                                                                              env_type, self_transitions)
    count_state_action = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch_traj:
        count_state_action[state, action] += 1

    # COMPUTE pi_star
    pi_star_old = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, current_proba, r_reshaped, 'default',
                              nb_traj=nb_trajectories)
    t_0 = time.time()
    pi_star_old.fit()
    t_1 = time.time()
    pi_star_old_perf = spibb.policy_evaluation_exact(pi_star_old.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Pi_star_old_perf: {pi_star_old_perf}')

    # Compute pi_star with batch_rl_algorithms.pi_star
    pi_star_new = PiStar(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=trajectories,
                         R=reward_current, episodic=episodic, P=current_proba)
    t_0 = time.time()
    pi_star_new.fit()
    t_1 = time.time()
    pi_star_new_perf = spibb.policy_evaluation_exact(pi_star_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Pi_star_new_perf: {pi_star_new_perf}')

    # COMPUTE the RL policy
    rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, reward_model, 'default',
                     nb_traj=nb_trajectories)
    t_0 = time.time()
    rl.fit()
    t_1 = time.time()
    # Evaluates the RL policy performance
    perfrl = spibb.policy_evaluation_exact(rl.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Pi_rl_old_perf: {perfrl}')

    # Compute basic_rl with batch_rl_algorithms.basic_rl
    rl_new = Basic_rl(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=trajectories,
                      R=reward_current, episodic=episodic)
    t_0 = time.time()
    rl_new.fit()
    t_1 = time.time()
    pi_rl_perf = spibb.policy_evaluation_exact(rl_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Pi_rl_new_perf: {pi_rl_perf}')

    N_wedge = 10

    # COMPUTE the Pi_b_SPIBB policy:
    mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch_traj)

    spibb_old = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model,
                            'Pi_b_SPIBB', nb_traj=nb_trajectories)
    t_0 = time.time()
    spibb_old.fit()
    t_1 = time.time()
    spibb_old_perf = spibb.policy_evaluation_exact(spibb_old.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'SPIBB_old: {spibb_old_perf}')

    # Compute spibb_b with batch_rl_algorithms.spibb
    spibb_new = SPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=trajectories,
                      R=reward_current, N_wedge=N_wedge, episodic=episodic)
    t_0 = time.time()
    spibb_new.fit()
    t_1 = time.time()
    spibb_new_perf = spibb.policy_evaluation_exact(spibb_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'SPIBB_new: {spibb_new_perf}')

    # COMPUTE the Pi_<b_SPIBB policy:
    pi_leq_b_SPIBB_old = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions,
                                     reward_model, 'Pi_leq_b_SPIBB', nb_traj=nb_trajectories)
    t_0 = time.time()
    pi_leq_b_SPIBB_old.fit()
    t_1 = time.time()
    # Evaluates the Pi_<b_SPIBB performance:
    perf_Pi_leq_b_SPIBB_old = \
        spibb.policy_evaluation_exact(pi_leq_b_SPIBB_old.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Lower_SPIBB_old: {perf_Pi_leq_b_SPIBB_old}')

    # Compute lower_spibb with batch_rl_algorithms.spibb
    lower_spibb_new = Lower_SPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=trajectories,
                                  R=reward_current, N_wedge=N_wedge, episodic=episodic)
    t_0 = time.time()
    lower_spibb_new.fit()
    t_1 = time.time()
    lower_spibb_new_perf = spibb.policy_evaluation_exact(lower_spibb_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Lower_SPIBB_new: {lower_spibb_new_perf}')

    # COMPUTE the R_min policy:
    N_wedge = 3
    # Computation of the binary mask for the bootstrapped state actions
    mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch_traj)

    pi_r_min = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model,
                           'R_min', nb_traj=nb_trajectories, q_pi_b_est=q_pi_b_est)
    pi_r_min.fit_advantageous()
    # Evaluates the Pi_r_min performance:
    perf_Pi_r_min = spibb.policy_evaluation_exact(pi_r_min.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'R_min_old: {perf_Pi_r_min}')

    # Compute R_min with batch_rl_algorithms.r_min:
    r_min = RMin(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=trajectories,
                 R=reward_current,
                 N_wedge=N_wedge, episodic=episodic)
    r_min.fit()
    r_min_new_perf = spibb.policy_evaluation_exact(r_min.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'R_min_new: {r_min_new_perf}')

    epsilon = 0.01;
    delta = 1
    # Computation of the binary mask for the bootstrapped state actions
    mask = None
    # Computation of the transition errors (e_q)
    errors_hoeffding = spibb.compute_errors(nb_states, nb_actions, delta, batch_traj)
    errors_mbie = spibb.compute_errors_q_mpeb(nb_states, nb_actions, delta, batch_traj, gamma)
    errors_min = np.minimum(errors_hoeffding, errors_mbie)
    # COMPUTE the Soft-SPIBB-sort-Q policy with hoeffding
    soft_SPIBB_sort_Q = spibb.spibb(
        gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_sort_Q',
        errors=errors_min, epsilon=epsilon, nb_traj=nb_trajectories
    )
    t_0 = time.time()
    soft_SPIBB_sort_Q.fit()
    t_1 = time.time()
    # Evaluates the Soft-SPIBB-sort-Q performance
    perf_soft_SPIBB_sort_Q = \
        spibb.policy_evaluation_exact(soft_SPIBB_sort_Q.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'approx_soft_spibb_old: {perf_soft_SPIBB_sort_Q}')

    # Compute ApproxSoftSPIBB with batch_rl_algorithms.soft_spibb
    approx_soft_spibb = ApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                        data=trajectories, R=reward_current, delta=delta, epsilon=epsilon,
                                        error_kind='min', episodic=episodic, checks=False)
    approx_soft_spibb.fit()
    approx_soft_spibb_new_perf = \
        spibb.policy_evaluation_exact(approx_soft_spibb.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'approx_soft_spibb_new: {approx_soft_spibb_new_perf}')

    # COMPUTE the Soft-SPIBB-sort-Q policy with hoeffding
    exact_soft_spibb_old = spibb.spibb(
        gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Soft_SPIBB_simplex',
        errors=errors_min, epsilon=epsilon, nb_traj=nb_trajectories
    )
    t_0 = time.time()
    exact_soft_spibb_old.fit()
    t_1 = time.time()
    # Evaluates the Soft-SPIBB-sort-Q performance
    perf_exact_soft_spibb_old = \
        spibb.policy_evaluation_exact(exact_soft_spibb_old.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'exact_soft_spibb_old: {perf_exact_soft_spibb_old}')

    # Compute ApproxSoftSPIBB with batch_rl_algorithms.soft_spibb
    exact_soft_spibb_new = ExactSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions,
                                          data=trajectories, episodic=episodic,
                                          R=reward_current, delta=delta, epsilon=epsilon, error_kind='min')
    exact_soft_spibb_new.fit()
    exact_soft_spibb_new_perf = \
        spibb.policy_evaluation_exact(exact_soft_spibb_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'exact_soft_spibb_new: {exact_soft_spibb_new_perf}')

    # COMPUTE the Lower-Soft-SPIBB-sort-Q policy with e_min
    lower_soft_SPIBB_sort_Q_e_min = spibb.spibb(
        gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model,
        'Lower_Soft_SPIBB_sort_Q', errors=errors_hoeffding, epsilon=epsilon, nb_traj=nb_trajectories
    )
    lower_soft_SPIBB_sort_Q_e_min.fit_advantageous()
    # Evaluates the Soft-SPIBB-sort-Q performance
    perf_lower_soft_SPIBB_sort_Q_e_min = \
        spibb.policy_evaluation_exact(lower_soft_SPIBB_sort_Q_e_min.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'lower_approx_soft_spibb_old: {perf_lower_soft_SPIBB_sort_Q_e_min}')

    # Compute LowerApproxSoftSPIBB with batch_rl_algorithms.soft_spibb
    lower_approx_soft_spibb_new = LowerApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states,
                                                       nb_actions=nb_actions,
                                                       data=trajectories, episodic=episodic,
                                                       R=reward_current, delta=delta, epsilon=epsilon,
                                                       error_kind='hoeffding')
    lower_approx_soft_spibb_new.fit()
    lower_approx_soft_spibb_new_perf = \
        spibb.policy_evaluation_exact(lower_approx_soft_spibb_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'lower_approx_soft_spibb_new: {lower_approx_soft_spibb_new_perf}')

    # COMPUTE the Adv-Soft-SPIBB-sort-Q policy with e_hoeffding
    adv_soft_SPIBB_sort_Q = spibb.spibb(
        gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model,
        'Adv_Approx_SBIBB_sort_Q', q_pi_b_est=q_pi_b_est,
        errors=errors_hoeffding, epsilon=epsilon, nb_traj=nb_trajectories
    )
    adv_soft_SPIBB_sort_Q.fit_advantageous()
    # Evaluates Advantageous-Soft-SPIBB-sort-Q performance with Hoeffding
    perf_adv_soft_SPIBB_sort_Q = \
        spibb.policy_evaluation_exact(adv_soft_SPIBB_sort_Q.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'adv_approx_soft_spibb_old: {perf_adv_soft_SPIBB_sort_Q}')

    # Compute AdvApproxSoftSPIBB with batch_rl_algorithms.soft_spibb
    adv_approx_soft_spibb_new = AdvApproxSoftSPIBB(pi_b=pi_b, gamma=gamma, nb_states=nb_states,
                                                   nb_actions=nb_actions,
                                                   data=trajectories, episodic=episodic,
                                                   R=reward_current, delta=delta, epsilon=epsilon,
                                                   error_kind='hoeffding')
    adv_approx_soft_spibb_new.fit()
    adv_approx_soft_spibb_new_perf = \
        spibb.policy_evaluation_exact(adv_approx_soft_spibb_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Adv_approx_soft_spibb_new: {adv_approx_soft_spibb_new_perf}')

    # DUIPI
    mask = None
    reward_model_full = np.zeros((nb_states, nb_actions, nb_states))
    for action in range(nb_actions):
        reward_model_full[:, action, :] = reward_current.copy()
    bayesian = True
    variance_P = np.zeros([nb_states, nb_actions, nb_states])
    if bayesian:
        alpha = 0.1
        count_state_action_state = np.zeros((nb_states, nb_actions, nb_states))
        for [action, state, next_state, reward] in batch_traj:
            count_state_action_state[state, action, next_state] += 1
        alpha_d = (count_state_action_state + alpha)
        alpha_d_0 = np.sum(alpha_d, 2)[:, :, np.newaxis]
        transitions = alpha_d / alpha_d_0
        variance_P = alpha_d * (alpha_d_0 - alpha_d) / alpha_d_0 ** 2 / (alpha_d_0 + 1)
    else:
        count_state_action = np.zeros((nb_states, nb_actions))
        for [action, state, next_state, reward] in batch_traj:
            count_state_action[state, action] += 1
        transitions = model.transitions
        for state in range(nb_states):
            variance_P[:, :, state] = transitions[:, :, state] * (1 - transitions[:, :, state]) / (
                    count_state_action - 1)
        variance_P = np.nan_to_num(variance_P, nan=1 / 4, posinf=1 / 4)  # maximal variance is (b - a)^2 / 4
        variance_P[
            count_state_action == 0] = 1 / 4  # Otherwise variance_P would be if a state-action pair hasn't been visited yet

    variance_R = np.zeros([nb_states, nb_actions, nb_states])

    param_duipi = 0

    # COMPUTE DUIPI
    duipi = spibb.spibb(
        gamma, nb_states, nb_actions, pi_b, mask, transitions, reward_model_full, 'DUIPI',
        param=param_duipi, nb_traj=nb_trajectories, q_pi_b_est=q_pi_b_est, variance_P=variance_P,
        variance_R=variance_R, range_rewards=4
    )
    duipi.fit_advantageous()
    # Evaluates the Soft-SPIBB-simplex performance
    perf_duipi = \
        spibb.policy_evaluation_exact(duipi.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'DUIPI_old {perf_duipi}')

    # Compute DUIPI with batch_rl_algorithms.duipi
    duipi_new = DUIPI(pi_b=pi_b, gamma=gamma, nb_states=nb_states,
                      nb_actions=nb_actions, data=trajectories, episodic=episodic,
                      R=reward_current, bayesian=bayesian, xi=param_duipi)
    duipi_new.fit()
    duipi_new_perf = spibb.policy_evaluation_exact(duipi_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'DUIPI_new: {duipi_new_perf}')

    kappa = 0
    # COMPUTE the Reward-adjusted MDP RL policy: (is the same as MBIE-EB)
    ramdp_reward_model = reward_model - kappa / np.sqrt(count_state_action)
    ramdp = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions,
                        ramdp_reward_model, 'default')
    ramdp.fit()
    # Evaluates the Reward-adjusted MDP RL policy performance
    perf_RaMDP = spibb.policy_evaluation_exact(ramdp.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'RaMDP_old: {perf_RaMDP}')
    # Compute RaMDP with batch_rl_algorithms.ramdp
    ramdp_new = RaMDP(pi_b=pi_b, gamma=gamma, nb_states=nb_states,
                      nb_actions=nb_actions, data=trajectories, episodic=episodic,
                      R=reward_current, kappa=kappa)
    ramdp_new.fit()
    ramdp_new_perf = spibb.policy_evaluation_exact(ramdp_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'RaMDP_new: {ramdp_new_perf}')

    delta = 0.9
    # COMPUTE the MBIE policy:
    # Computation of the binary mask for the unseen state actions
    mask_unseen = ~spibb.compute_mask_N_wedge(nb_states, nb_actions, 1, batch_traj)
    count_state_action = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch_traj:
        count_state_action[state, action] += 1
    errors_mpeb = np.zeros([nb_states, nb_actions])
    errors_mpeb_original = np.zeros([nb_states, nb_actions])
    for state in range(nb_states):
        for action in range(nb_actions):
            errors_mpeb[state, action] = np.sqrt(
                2 * (np.log(2 * (nb_states * nb_actions) * 2 ** nb_states / delta)) /
                count_state_action[
                    state, action]
            )

            errors_mpeb_original[state, action] = np.sqrt(
                2 * (np.log(2 * (nb_states * nb_actions) * (2 ** nb_states - 2) / delta)) /
                count_state_action[
                    state, action]
            )
    errors_mpeb[count_state_action == 0] = 2  # Maximal L1 distance between to prob. dists
    errors_mpeb_original[count_state_action == 0] = 2  # Maximal L1 distance between to prob. dists
    pi_mbie = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_unseen, model.transitions,
                          reward_model_full,
                          'MBIE', nb_traj=nb_trajectories, q_pi_b_est=q_pi_b_est,
                          mask_unseen=mask_unseen,
                          errors=errors_mpeb_original)
    t_0 = time.time()
    pi_mbie.fit_advantageous()
    t_1 = time.time()
    # Evaluates the MBIE performance:
    perf_Pi_mbie = spibb.policy_evaluation_exact(pi_mbie.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'Perf_mbie: {perf_Pi_mbie}')

    # Compute MBIE with batch_rl_algorithms.mbie
    mbie_new = MBIE(pi_b=pi_b, gamma=gamma, nb_states=nb_states,
                    nb_actions=nb_actions, data=trajectories, episodic=episodic,
                    R=reward_current, delta=delta)
    mbie_new.fit()
    mbie_new_perf = spibb.policy_evaluation_exact(mbie_new.pi, r_reshaped, current_proba, gamma)[0][0]
    print(f'MBIE_new: {mbie_new_perf}')
#
#    advantage_pi_b = spibb_utils.get_advantage(0, q_pi_b_est, pi_b, pi_b)
#    advantage_r_min = spibb_utils.get_advantage(0, q_pi_b_est, pi_b, pi_r_min.pi)
#    advantage_basic_rl = spibb_utils.get_advantage(0, q_pi_b_est, pi_b, rl.pi)
#    advantage_pi_star = spibb_utils.get_advantage(0, q_pi_b_est, pi_b, pi_star_old.pi)
#
#    print(f'Performance DUIPI: {perf_Pi_r_min}')
#
