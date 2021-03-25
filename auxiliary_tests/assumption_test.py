import os
import sys
import numpy as np
import pandas as pd
import configparser

directory = os.path.dirname(os.path.expanduser(__file__))
sys.path.append(directory)
path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'paths.ini'))
spibb_path = path_config['PATHS']['spibb_path']
sys.path.append(spibb_path)

import garnets
import spibb_utils
import spibb

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

def compute_errors_p(nb_states, nb_actions, delta, batch_traj, unvisited=np.inf):
    count_state_action = np.zeros((nb_states, nb_actions))
    errors = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch_traj:
        count_state_action[state, action] += 1
    for state in range(nb_states):
        for action in range(nb_actions):
            if count_state_action[state, action] == 0:
                errors[state, action] = unvisited
            else:
                errors[state, action] = np.sqrt(
                    2*(np.log(2*(nb_states*nb_actions*2**nb_actions)/delta))/count_state_action[state, action]
                )
    return errors

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
        # These are the e_q
        # errors = spibb.compute_errors(nb_states, nb_actions, delta, batch_traj)
        errors = compute_errors_p(nb_states, nb_actions, delta, batch_traj, unvisited=2)
        # Terminal state has no errors/insecurities (also before it is 'inf' for the terminal state)
        errors[34, :] = np.zeros(4)

        kappa = 1 / gamma  # ~1.053

        expected_errors = np.zeros([nb_states, nb_actions])

        for s0 in range(nb_states):
            for a0 in range(nb_actions):
                if errors[s0, a0] != 0:
                    for s1 in range(nb_states):
                        for a1 in range(nb_actions):
                            expected_errors[s0, a0] += errors[s1, a1] * pi_b[s1, a1] * current_proba[s0, a0, s1]

        assumption_holds = expected_errors <= kappa * errors
        nb_state_action_pairs = nb_actions * nb_states
        failed = nb_state_action_pairs - sum(sum(assumption_holds))
        results_traj.append(failed)
        print(
            f'For ratio {ratio} and {nb_trajectories} trajectories, {failed} out of '
            f'{nb_state_action_pairs} state_action pairs failed Assumption 1. ')
            # f'(Unvisited state action pairs: {nb_not_visited_state_action_pairs})')
    results.append(results_traj)


results_df = pd.DataFrame(index=ratios, columns=nb_trajectories_list, data=results)
import seaborn as sns
sns.heatmap(results_df)
