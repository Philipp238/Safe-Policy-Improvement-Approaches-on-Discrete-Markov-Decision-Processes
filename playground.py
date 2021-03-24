import os
import sys
import numpy as np
import pandas as pd

sys.path.append(r'C:\Users\phili\PycharmProjects\SPIBB')

import garnets
import spibb_utils
import spibb
import modelTransitions

nb_trajectories_list = [10, 20, 50, 100, 200, 500, 1000, 2000]
delta = 0.05
epsilons = [0.1]  # [0.1, 0.2, 0.5, 1, 2, 5], value of epsilon not interesting for us
ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# For debugging purposes
nb_trajectories = 5000
epsilon = 0.5
ratio = 0.7
# Ending

seed = 1234
np.random.seed(seed)
gamma = 0.95
nb_states = 50
nb_actions = 4
nb_next_state_transition = 4
env_type = 1  # 1 for one terminal state, 2 for two terminal states

self_transitions = 0


def compute_errors_p(nb_states, nb_actions, delta, batch_traj):
    count_state_action = np.zeros((nb_states, nb_actions))
    errors = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch_traj:
        count_state_action[state, action] += 1
    for state in range(nb_states):
        for action in range(nb_actions):
            if count_state_action[state, action] == 0:
                errors[state, action] = np.inf
            else:
                errors[state, action] = np.sqrt(
                    2 * (np.log(2 * (nb_states * nb_actions * 2 ** nb_states) / delta)) / count_state_action[
                        state, action]
                )
    return errors


def compute_errors_q_mpeb(nb_states, nb_actions, delta, batch_traj, gamma):
    # Compute the MPeB
    # 1. Start with the samples for q for each state action pair
    discounted_reward = 1
    count_state_action = np.zeros([nb_states, nb_actions])
    q_samples = np.empty([nb_states, nb_actions], dtype=object)
    for x in range(nb_states):
        for a in range(nb_actions):
            q_samples[x, a] = np.array([])
    for [action, state, next_state, reward] in reversed(batch_traj):
        if reward == 1:
            discounted_reward = 1
        else:
            discounted_reward *= gamma
        count_state_action[int(state), action] += 1
        q_samples[state, action] = np.append(q_samples[state, action], discounted_reward)

    # 2. Get the estimate for Q
    Q_batch = np.zeros([nb_states, nb_actions])
    for state in range(nb_states):
        for action in range(nb_actions):
            Q_batch[state, action] = np.mean(q_samples[state, action])

    # 3. Get the variance of the estimate
    var_q = np.zeros([nb_states, nb_actions])
    var_q_np = np.zeros([nb_states, nb_actions])
    for state in range(nb_states):
        for action in range(nb_actions):
            var_q_np[state, action] = np.var(q_samples[state, action], ddof=1)

    # 4. Compute the MPeB errors
    e_mpeb = np.zeros([nb_states, nb_actions])
    for state in range(nb_states):
        for action in range(nb_actions):
            e_mpeb[state, action] = 2 * (np.sqrt(
                2 * var_q[state, action] * np.log(4 * nb_states * nb_actions / delta) / count_state_action[
                    state, action]) + 7 * np.log(
                4 * nb_states * nb_actions / delta) / (3 * (count_state_action[state, action] - 1)))

    e_mpeb = np.nan_to_num(e_mpeb, nan=np.inf, posinf=np.inf)
    return e_mpeb


def compute_q_by_mc(nb_states, nb_actions, delta, batch_traj):
    discounted_reward = 1
    count_state_action = np.zeros([nb_states, nb_actions])
    Q = np.zeros([nb_states, nb_actions])
    for [action, state, next_state, reward] in reversed(batch_traj):
        if reward == 1:
            discounted_reward = 1
        else:
            discounted_reward *= gamma
        count_state_action[int(state), action] += 1
        Q[int(state), action] += 1 / count_state_action[int(state), action] * (
                discounted_reward - Q[int(state), action])
    return Q


e_2 = compute_errors_q_mpeb(nb_states, nb_actions, delta, batch_traj, gamma)

for ratio in ratios:
    garnet = garnets.Garnets(nb_states, nb_actions, nb_next_state_transition,
                             env_type=env_type, self_transitions=self_transitions)

    softmax_target_perf_ratio = (ratio + 1) / 2
    baseline_target_perf_ratio = ratio
    pi_b, q_pi_b, pi_star_perf, pi_b_perf, pi_rand_perf = \
        garnet.generate_baseline_policy(gamma,
                                        softmax_target_perf_ratio=softmax_target_perf_ratio,
                                        baseline_target_perf_ratio=baseline_target_perf_ratio)

    reward_current = garnet.compute_reward()
    current_proba = garnet.transition_function
    r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)

    for nb_trajectories in nb_trajectories_list:
        # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
        trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b)

        # Understand how the transitions model got calculated
        count_state_action_next = np.zeros((nb_states, nb_actions, nb_states))
        for [action, state, next_state, _] in batch_traj:
            count_state_action_next[int(state), action, int(next_state)] += 1
        transitions = count_state_action_next / np.sum(count_state_action_next, 2)[:, :, np.newaxis]
        transitions = np.nan_to_num(transitions)

        # How is q calculated?
        pi = pi_b
        P = transitions
        R = r_reshaped.reshape(nb_states * nb_actions)

        nb_sa = nb_states * nb_actions
        M = np.eye(nb_sa) - gamma * np.einsum('ijk,kl->ijkl', P, pi).reshape(nb_sa, nb_sa)
        q = np.dot(np.linalg.inv(M), R).reshape(nb_states, nb_actions)

        # Analyse einsum
        m = np.einsum('ijk,kl->ijkl', P, pi)
        i, j, k, l = 0, 0, 6, 0
        P[i, j, k] * pi[k, l] == m[i, j, k, l]  # TRUE

        # Now compute Q by monte carlo estimation (sample average)
        discounted_reward = 1
        count_state_action = np.zeros([nb_states, nb_actions])
        Q = np.zeros([nb_states, nb_actions])
        for [action, state, next_state, reward] in reversed(batch_traj):
            if reward == 1:
                discounted_reward = 1
            else:
                discounted_reward *= gamma
            count_state_action[int(state), action] += 1
            Q[int(state), action] += 1 / count_state_action[int(state), action] * (
                    discounted_reward - Q[int(state), action])

        # Compute the MPeB
        # 1. Start with the samples for q for each state action pair
        discounted_reward = 1
        count_state_action = np.zeros([nb_states, nb_actions])
        q_samples = np.empty([nb_states, nb_actions], dtype=object)
        for x in range(nb_states):
            for a in range(nb_actions):
                q_samples[x, a] = np.array([])
        for [action, state, next_state, reward] in reversed(batch_traj):
            if reward == 1:
                discounted_reward = 1
            else:
                discounted_reward *= gamma
            count_state_action[int(state), action] += 1
            q_samples[state, action] = np.append(q_samples[state, action], discounted_reward)

        # 2. Get the estimate for Q
        Q_batch = np.zeros([nb_states, nb_actions])  # Luckily, we see that both calculations of Q are equivalent
        for state in range(nb_states):
            for action in range(nb_actions):
                Q_batch[state, action] = np.mean(q_samples[state, action])

        # 3. Get the variance of the estimate
        var_q = np.zeros([nb_states, nb_actions])
        var_q_np = np.zeros([nb_states, nb_actions])
        for state in range(nb_states):
            for action in range(nb_actions):
                var_q_np[state, action] = np.var(q_samples[state, action], ddof=1)

        # Alternative for 1-3: Compute the variance iteratively (gives the same as the batch program)
        SS_q_iter = np.zeros([nb_states, nb_actions])
        q_iter = np.zeros([nb_states, nb_actions])
        discounted_reward = 1
        count_state_action = np.zeros([nb_states, nb_actions])
        for [action, state, next_state, reward] in reversed(batch_traj):
            if reward == 1:
                discounted_reward = 1
            else:
                discounted_reward *= gamma
            count_state_action[int(state), action] += 1
            old_q_iter = q_iter[int(state), action]
            q_iter[int(state), action] += 1 / count_state_action[int(state), action] * (
                    discounted_reward - q_iter[int(state), action])
            SS_q_iter[int(state), action] += (discounted_reward - q_iter[int(state), action]) * (
                    discounted_reward - old_q_iter)
        var_q_iter = SS_q_iter / (count_state_action - 1)

        # 4. Compute the MPeB errors
        e_mpeb = np.zeros([nb_states, nb_actions])
        for state in range(nb_states):
            for action in range(nb_actions):
                e_mpeb[state, action] = 2 * (np.sqrt(
                    2 * var_q[state, action] * np.log(4 * nb_states * nb_actions / delta) / count_state_action[
                        state, action]) + 7 * np.log(
                    4 * nb_states * nb_actions / delta) / (3 * (count_state_action[state, action] - 1)))

        e_mpeb = np.nan_to_num(e_mpeb, nan=np.inf, posinf=np.inf)
        # Computation of the transition errors
        # These are the e_q
        errors = spibb.compute_errors(nb_states, nb_actions, delta, batch_traj)
        # Compare the errors
        e_min = np.minimum(e_mpeb, errors)
        np.median(e_mpeb)
        np.median(errors)
        np.median(e_min)
        sum(e_mpeb < errors)
        e_mpeb_0 = np.nan_to_num(e_mpeb, posinf=0)
        errors_0 = np.nan_to_num(errors, posinf=0)
        e_min_0 = np.nan_to_num(e_min, posinf=0)
        sum(e_mpeb_0)
        sum(errors_0)
        sum(e_min_0)

        pd.DataFrame(e_mpeb_0).plot(kind='density', bw_method=1)
        pd.DataFrame(errors_0).plot(kind='density', bw_method=1)
        pd.DataFrame(e_min_0).plot(kind='density', bw_method=1)
        import seaborn as sns

        sns.displot(e_mpeb.flatten()[np.isfinite(e_mpeb.flatten())])
        sns.displot(errors.flatten()[np.isfinite(errors.flatten())])
        sns.displot(e_mpeb.flatten()[np.isfinite(e_mpeb.flatten())])
        for epsilon in epsilons:

            count_state_action = np.zeros((nb_states, nb_actions))
            for [action, state, next_state, reward] in batch_traj:
                count_state_action[state, action] += 1

            # errors = compute_errors_p(nb_states, nb_actions, delta, batch_traj)
            # Terminal state has no errors/insecurities (also before it is 'inf' for the terminal state)
            errors[34, :] = np.zeros(4)
            nb_not_visited_state_action_pairs = sum(sum(errors == np.inf))
            errors[errors == np.inf] = 0  # very conservative, to avoid problems with inf

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
            print(
                f'For ratio {ratio} and {nb_trajectories} trajectories, {failed} out of '
                f'{nb_state_action_pairs} state_action pairs failed Assumption 1. '
                f'(Unvisited state action pairs: {nb_not_visited_state_action_pairs})')


# error_difference = expected_errors - kappa * errors

# errors.shape  # (50, 4): errors[s, a] = error of state-action pair (s, a)
# pi_b.shape  # (50,4): pi_b[s, a] = probability of choosing action a when in state s under the baseline policy

# print(reward_current.shape)  # (50, 50)
# print(current_proba.shape)  # (50, 4, 50)

# sum(current_proba[0, 0, :])  # equals 1
# ==> current_proba[s0, a, s1] Probability of ending up in state s1 if the MDP is in state s0 and action a is chosen

# batch_traj[i] = [action_choice, state, next_state, reward]


def calc_N_wedge():
    x = 50
    a = 4
    delta = 0.05
    gamma = 0.95
    N_wedges = [5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 50000, 10 ** 5, 5 * 10 ** 5, 10 ** 6, 5 * 10 ** 6]
    for N_wedge in N_wedges:
        epsilon = 4 / (1 - gamma) * np.sqrt(2 / N_wedge * np.log(2 * x * a * 2 ** x / delta))
        print(f'N_wedge {N_wedge}: epsilon {epsilon}')


x = np.arange(24).reshape([2, 3, 4])
y = np.arange(6).reshape([2, 3])
z = np.arange(12).reshape([3, 4])
y_strided = np.lib.stride_tricks.as_strided(y, shape=(2, 3, 4), strides=(12,4,0))
x / y
x / z

x.dtype
