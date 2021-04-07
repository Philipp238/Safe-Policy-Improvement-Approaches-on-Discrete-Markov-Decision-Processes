# File to check that the two different action-value functions (MC estimate and the action-value function in the
# estimated MDP) are actually different functions, see Section 3.2.2 in "Evaluation of Safe Policy Improvement with
# Soft Baseline Bootstrapping" by Philipp Scholl.
import os
import sys
import numpy as np
import pandas as pd
import configparser

# Set directory as the path to Evaluation-of-Safe-Policy-Improvement-with-Baseline-Bootstrapping
# directory = os.path.dirname(os.path.dirname(os.path.expanduser(__file__)))
directory = r'C:\Users\phili\PycharmProjects\Evaluation-of-Safe-Policy-Improvement-with-Baseline-Bootstrapping'

sys.path.append(directory)
path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'paths.ini'))
spibb_path = path_config['PATHS']['spibb_path']
sys.path.append(spibb_path)

import garnets
import spibb_utils
import spibb
import modelTransitions

nb_trajectories_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
delta = 0.05
ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

seed = 1234
np.random.seed(seed)
gamma = 0.7
nb_states = 50
nb_actions = 4
nb_next_state_transition = 4
env_type = 1  # 1 for one terminal state, 2 for two terminal states

self_transitions = 0


results = []

for ratio in ratios:
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
    r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)
    results_traj = []

    for nb_trajectories in nb_trajectories_list:
        # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
        trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b)

        # Computation of the transition errors

        model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
        reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)
        # q_pi_b_est is the MC estimation of the action-value function
        q_pi_b_est = spibb_utils.compute_q_pib_est_episodic(gamma=gamma, nb_actions=nb_actions, nb_states=nb_states,
                                                            batch=trajectories)
        # q_m_hat is the action-value function in the estimated MDP.
        _, q_m_hat = spibb.policy_evaluation_exact(pi_b, reward_model, model.transitions, gamma)

        distance = np.linalg.norm(q_pi_b_est - q_m_hat, ord=1)

        results_traj.append(distance)
        print(
            f'For ratio {ratio} and {nb_trajectories} trajectories, the L1 distance in the two calculations of q of '
            f'pi_b is {distance}.')
            # f'(Unvisited state action pairs: {nb_not_visited_state_1action_pairs})')
    results.append(results_traj)
