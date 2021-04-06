import numpy as np
from scipy.optimize import linprog

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class SoftSPIBB(BatchRLAlgorithm):
    # Abstract base class for all Soft-SPIBB algorithms.
    NAME = 'abstract_soft_spibb_class'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, error_kind, delta, epsilon, episodic,
                 zero_unseen=True, max_nb_it=5000, checks=False, speed_up_dict=None, g_max=None,
                 ensure_independence=False, allowed_correlation=0.01):
        """
        :param pi_b: numpy matrix with shape (nb_states, nb_actions), such that pi_b(s,a) refers to the probability of
        choosing action a in state s by the behavior policy
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: the data collected by the behavior policy, which should be a list of [state, action, next_state,
         reward] sublists
        :param R: reward matrix with shape (nb_states, nb_states), assuming that the reward is deterministic w.r.t. the
         previous and the next states
        :param episodic: boolean variable, indicating whether the MDP is episodic (True) or non-episodic (False)
        :param zero_unseen: boolean variable, indicating whether the estimated model should guess set all transition
        probabilities to zero for a state-action pair which has never been visited (True) or to 1/nb_states (False)
        :param max_nb_it: integer, indicating the maximal number of times the PE and PI step should be executed, if
        convergence is not reached
        :param checks: boolean variable indicating if different validity checks should be executed (True) or not
        (False); this should be set to True for development reasons, but it is time consuming for big experiments
        :param speed_up_dict: a dictionary containing pre-calculated quantities which can be reused by many different
        algorithms, this should only be used for big experiments; for Approx-Soft-SPIBB, Exact-Soft-SPIBB and
        Lower-Approx-Soft-SPIBB using error_kind='hoeffding' this should only contain the following:
            'count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of times a s
            tate-action pair has been visited
            'count_state_action_state': numpy array with shape (nb_states, nb_actions, nb_states) indicating the number
            of times a state-action-next-state triplet has been visited
        If ensure_indepence=True it should contain additionally:
            'augmented_count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of
            times a state-action pair has been visited, after all returns were omitted such that the correlation is
            below allowed_correlation
        For the 1-Step versions and Advantageous-Approx-Soft-SPIBB one has to add to the original two elements:
            'q_pi_b_est': monte carlo estimate of the action-value function as numpy array with shape (nb_states,
            nb_actions)
        For any Soft-SPIBB algorithm using error_kind='mpeb' one has to add to the two original elements the following:
            'q_pi_b_est': monte carlo estimate of the action-value function as numpy array with shape (nb_states,
            nb_actions)
            'var_q': estimation of the variance of the monte carloe estimate fo the action-value function as numpy array
            with shape (nb_states, nb_actions)
        :param error_kind: String stating which error function should be used, 'hoeffding' and 'mpeb' are available
        :param delta: hyper-parameter for all Soft-SPIBB algorithms
        :param epsilon: hyper-parameter for all Soft-SPIBB algorithms
        :param g_max: maximal return of the (centralized) MDP, only necessary if error_kind='mpeb'
        :param ensure_independence: boolean variable indicating if some data should be ommited to ensure that the
        correlation between two returns of the state-action pair does not surpass the allowed_correlation
        :param allowed_correlation: positive float which is only necessary if ensure_independence is true and gives the
        upper bound on the correlation between two return for one state-action pair in that case
        """
        self.delta = delta
        self.epsilon = epsilon
        self.error_kind = error_kind
        self.ensure_independence = ensure_independence
        if self.ensure_independence:
            self.allowed_correlation = allowed_correlation
            self.minimum_steps_between_q_samples = np.ceil(np.log(self.allowed_correlation) / np.log(gamma))
        if error_kind == 'mpeb' and not g_max:
            raise AssertionError('You have specified error_kind=\'mpeb\', but did not provide g_max.')
        if episodic and ensure_independence:
            raise NotImplementedError('You have specified epsiodic=True, but used ensure_independence=True.')
        self.g_max = g_max
        super().__init__(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=data, R=R,
                         zero_unseen=zero_unseen, max_nb_it=max_nb_it, episodic=episodic, checks=checks,
                         speed_up_dict=speed_up_dict)
        self.old_pi = None

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        super()._initial_calculations()
        if not self.ensure_independence:
            self.augmented_count_state_action = self.count_state_action
        self._compute_errors()

    def _compute_errors(self):
        """
        Starts the computation of the error function.
        """
        if self.error_kind == 'hoeffding':
            self.errors = self._compute_hoeffding_errors()
        elif self.error_kind == 'mpeb':
            self.errors = self._compute_mpeb_errors()
        else:
            raise AssertionError(
                f'No error method named {self.error_kind} available, please choose among \'hoeffding\' and \'mpeb\'!')

    # Computes the errors in q for all state-action pairs using hoeffding
    def _compute_hoeffding_errors(self):
        """
        Computes the error function for all state-action pairs using Hoeffding's bound.
        """
        errors = np.zeros((self.nb_states, self.nb_actions))
        if self.ensure_independence:
            if self.speed_up_dict:
                self.augmented_count_state_action = self.speed_up_dict['augmented_count_state_action']
            else:
                self.augmented_count_state_action = np.zeros((self.nb_states, self.nb_actions))
                last_time_step = - np.ones((self.nb_states, self.nb_actions)) * np.inf
                for i, [action, state, next_state, reward] in enumerate(self.data):
                    if i - last_time_step[state, action] > self.minimum_steps_between_q_samples:
                        self.augmented_count_state_action[state, action] += 1
                        last_time_step[state, action] = i
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                # TODO: Hoeffding errors to fit if self.ensure_independence=True
                if self.augmented_count_state_action[state, action] == 0:
                    errors[state, action] = np.inf
                else:
                    errors[state, action] = np.sqrt(
                        2 * (np.log(2 * (self.nb_states * self.nb_actions) / self.delta)) /
                        self.augmented_count_state_action[
                            state, action]
                    )
        return errors

    def _compute_q_pi_b_samples(self):
        """
        Computes all returns for each state-action pairs.
        """
        self.q_samples = np.empty([self.nb_states, self.nb_actions], dtype=object)
        if self.ensure_independence:
            self.q_samples_time_steps = np.empty([self.nb_states, self.nb_actions], dtype=object)
        for x in range(self.nb_states):
            for a in range(self.nb_actions):
                self.q_samples[x, a] = np.array([])
                if self.ensure_independence:
                    self.q_samples_time_steps[x, a] = np.array([])
        if self.episodic:
            for episode in self.data:
                discounted_reward = 0
                for [action, state, next_state, reward] in reversed(episode):
                    discounted_reward = self.gamma * discounted_reward + reward
                    self.q_samples[state, action] = np.append(self.q_samples[state, action], discounted_reward)
        else:
            discounted_reward = 0
            for i, [action, state, next_state, reward] in enumerate(reversed(self.data)):
                discounted_reward = self.gamma * discounted_reward + reward
                self.q_samples[state, action] = np.append(self.q_samples[state, action], discounted_reward)
                if self.ensure_independence:
                    self.q_samples_time_steps[state, action] = np.append(self.q_samples_time_steps[state, action],
                                                                         len(self.data) - i)
            if self.ensure_independence:
                self._discard_q_pi_samples_to_ensure_independence()

    def _discard_q_pi_samples_to_ensure_independence(self):
        """
        Discards specific returns to ensure that the correlation between two of the same sate-action pair is below
        self.allowed_correlation.
        """
        self.augmented_count_state_action = np.zeros((self.nb_states, self.nb_actions))
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                last_time_step = - np.inf
                mask = []
                for time_step in reversed(self.q_samples_time_steps[state, action]):
                    if time_step - last_time_step > self.minimum_steps_between_q_samples:
                        self.augmented_count_state_action[state, action] += 1
                        last_time_step = time_step
                        mask.insert(0, True)
                    else:
                        mask.insert(0, False)
                self.q_samples[state, action] = self.q_samples[state, action][mask]

    def _compute_q_pi_b_est_from_samples(self):
        """
        Computes the mean of all returns for each state-action pair.
        """
        self.q_pi_b_est = np.zeros([self.nb_states, self.nb_actions])
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                self.q_pi_b_est[state, action] = np.mean(self.q_samples[state, action])

    def _compute_var_q_pi_b_est(self):
        """
        Computes the variance of all returns for each state-action pair.
        """
        var_q = np.zeros([self.nb_states, self.nb_actions])
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                var_q[state, action] = np.var(self.q_samples[state, action], ddof=1)
        return var_q

    # Computes the errors in q for all state-action pairs using mpeb
    def _compute_mpeb_errors(self):
        """
        Computes the error function for all state-action pairs using Maurer and Pontil's bound.
        """
        if self.speed_up_dict:
            self.q_pi_b_est = self.speed_up_dict['q_pi_b_est']
            self.var_q = self.speed_up_dict['var_q']
            if self.ensure_independence:
                self.augmented_count_state_action = self.speed_up_dict['augmented_count_state_action']
        else:
            # 1. Start with the samples for q for each state action pair
            self._compute_q_pi_b_samples()
            # 2. Get the estimate for Q (save it as property as we will need this at other points as well)
            self._compute_q_pi_b_est_from_samples()
            # 3. Get the variance of the estimate
            self.var_q = self._compute_var_q_pi_b_est()
        scaled_variance_q = self.var_q * 1 / ((2 * self.g_max) ** 2)
        # 4. Compute the MPeB errors
        e_mpeb = np.zeros([self.nb_states, self.nb_actions])
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                e_mpeb[state, action] = 2 * (np.sqrt(
                    2 * scaled_variance_q[state, action] * np.log(4 * self.nb_states * self.nb_actions / self.delta) /
                    self.augmented_count_state_action[
                        state, action]) + 7 * np.log(
                    4 * self.nb_states * self.nb_actions / self.delta) / (
                                                     3 * (self.augmented_count_state_action[state, action] - 1)))
        e_mpeb = np.nan_to_num(e_mpeb, nan=np.inf, posinf=np.inf)
        return e_mpeb

    def _check_if_constrained_policy(self):
        constrained = True
        for state in range(self.nb_states):
            distance = 0
            for action in range(self.nb_actions):
                distance += np.abs(self.pi[state, action] - self.pi_b[state, action]) * self.errors[state, action]
            if distance > self.epsilon * (1 + 10 ** (-3)):
                constrained = False
        if not constrained:
            print(f'!!! The policy of {self.NAME} is not constrained !!!')

    def _check_if_advantageous_policy(self):
        advantageous = True
        for state in range(self.nb_states):
            advantage = self.get_advantage(state)
            if advantage < - 10 ** (-6):
                advantageous = False
        if not advantageous:
            print(f'!!! The policy of {self.NAME} is not advantageous !!!')

    def get_advantage(self, state):
        # Set all nan values to 0, as they play no role for Soft-SPIBB batch_rl_algorithms, because pi[s,a] = pi_b[s,a] if s,a
        # has never been visited
        q_pi_b_est_no_nan = np.nan_to_num(self.q_pi_b_est, nan=0)
        advantage = q_pi_b_est_no_nan[state] @ (self.pi[state] - self.pi_b[state])
        return advantage

    def _one_step_algorithms(self):
        """
        Switches the optimization problem for the 1-step algorithms such that they are also theoretical safe, as there
        is otherwise no reason to use only 1 iteration.
        """
        if self.max_nb_it == 1:
            if self.error_kind == 'hoeffding' and self.NAME != AdvApproxSoftSPIBB.NAME:
                if self.speed_up_dict:
                    self.q_pi_b_est = self.speed_up_dict['q_pi_b_est']
                else:
                    self._compute_q_pi_b_samples()
                    self._compute_q_pi_b_est_from_samples()
            self.q = np.nan_to_num(self.q_pi_b_est, nan=0)


class ApproxSoftSPIBB(SoftSPIBB):
    # Code and algorithm from 'Safe Policy Improvement with Soft Baseline Bootstrapping' by Kimia Nadjahi, Romain
    # Laroche and Remi Tachet des Combes. (Their code is available under: https://github.com/RomainLaroche/SPIBB)
    NAME = 'Approx-Soft-SPIBB'

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """
        self._one_step_algorithms()

        pi = np.zeros([self.nb_states, self.nb_actions])
        pi_t = self.pi_b.copy()
        for s in range(self.nb_states):
            A_bot = np.argsort(self.q[s, :])  # increasing order
            allowed_error = self.epsilon
            for a_bot in A_bot:
                mass_bot = min(pi_t[s, a_bot], allowed_error / (2 * self.errors[s, a_bot]))
                #  A_top is sorted in decreasing order :
                A_top = np.argsort(-(self.q[s, :] - self.q[s, a_bot]) / self.errors[s, :])
                for a_top in A_top:
                    if a_top == a_bot:
                        break
                    mass_top = min(mass_bot, allowed_error / (2 * self.errors[s, a_top]))
                    if mass_top > 0:
                        mass_bot -= mass_top
                        pi_t[s, a_bot] -= mass_top
                        pi_t[s, a_top] += mass_top
                        allowed_error -= mass_top * (self.errors[s, a_bot] + self.errors[s, a_top])
                        if mass_bot == 0:
                            break
            # Local policy improvement check, required for convergence
            if self.old_pi is not None:
                new_local_v = pi_t[s, :].dot(self.q[s, :])
                old_local_v = self.old_pi[s, :].dot(self.q[s, :])
                if new_local_v >= old_local_v:
                    pi[s] = pi_t[s]
                else:
                    pi[s] = self.old_pi[s]
            else:
                pi[s] = pi_t[s]
        self.pi = pi
        if self.checks:
            self._check_if_constrained_policy()
        self.old_pi = self.pi.copy()


class ExactSoftSPIBB(SoftSPIBB):
    # Code and algorithm from 'Safe Policy Improvement with Soft Baseline Bootstrapping' by Kimia Nadjahi, Romain
    # Laroche and Remi Tachet des Combes. (Their code is available under: https://github.com/RomainLaroche/SPIBB)
    NAME = 'Exact-Soft-SPIBB'

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """

        self._one_step_algorithms()

        pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            finite_err_idx = self.errors[s] < np.inf
            c = np.zeros(2 * self.nb_actions)
            c[0:self.nb_actions] = -self.q[s, :]
            Aeq = np.zeros(2 * self.nb_actions)
            Aeq[0:self.nb_actions] = 1
            Aub = np.zeros(2 * self.nb_actions)
            Aub[self.nb_actions:2 * self.nb_actions][finite_err_idx] = self.errors[s, finite_err_idx]
            Aeq = [Aeq]
            beq = [1]
            Aub = [Aub]
            bub = [self.epsilon]
            if finite_err_idx.sum() == 0:
                pi[s] = self.pi_b[s]
            else:
                for idx in range(len(finite_err_idx)):
                    if not finite_err_idx[idx]:
                        new_Aeq = np.zeros(2 * self.nb_actions)
                        new_Aeq[idx] = 1
                        Aeq.append(new_Aeq)
                        beq.append(self.pi_b[s, idx])
                    else:
                        new_Aub = np.zeros(2 * self.nb_actions)
                        new_Aub[idx] = 1
                        new_Aub[idx + self.nb_actions] = -1
                        Aub.append(new_Aub)
                        bub.append(self.pi_b[s, idx])
                        new_Aub_2 = np.zeros(2 * self.nb_actions)
                        new_Aub_2[idx] = -1
                        new_Aub_2[idx + self.nb_actions] = -1
                        Aub.append(new_Aub_2)
                        bub.append(-self.pi_b[s, idx])
                res = linprog(c, A_eq=Aeq, b_eq=beq, A_ub=Aub, b_ub=bub)
                pi[s] = [p if p >= 0 else 0.0 for p in res.x[0:self.nb_actions]]  # Fix rounding error
        self.pi = pi
        if self.checks:
            self._check_if_constrained_policy()


class LowerApproxSoftSPIBB(SoftSPIBB):
    # Implements Lower-Approx-Soft-SPIBB developed in 'Evaluation of Safe Policy Improvement by Soft Baseline
    # Bootstrapping' by Philipp Scholl.
    NAME = 'Lower-Approx-Soft-SPIBB'

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """

        old_pi = self.pi.copy()
        pi = np.zeros([self.nb_states, self.nb_actions])
        pi_t = self.pi_b.copy()
        for s in range(self.nb_states):
            A_bot = np.argsort(self.q[s, :])  # increasing order
            allowed_error = self.epsilon
            for a_bot in A_bot:
                mass_bot = pi_t[s, a_bot]
                #  A_top is sorted in decreasing order :
                A_top = np.argsort(-(self.q[s, :] - self.q[s, a_bot]) / self.errors[s, :])
                for a_top in A_top:
                    if a_top == a_bot:
                        break
                    # mass_top = min(mass_bot, allowed_error / (2 * self.errors[s, a_top]))  # Here only for comparison
                    mass_top = min(mass_bot, allowed_error / (1 * self.errors[s, a_top]))
                    if mass_top > 0:
                        mass_bot -= mass_top
                        pi_t[s, a_bot] -= mass_top
                        pi_t[s, a_top] += mass_top
                        allowed_error -= mass_top * self.errors[s, a_top]
                        if mass_bot == 0:
                            break
            # Local policy improvement check, required for convergence
            if old_pi is not None:
                new_local_v = pi_t[s, :].dot(self.q[s, :])
                old_local_v = old_pi[s, :].dot(self.q[s, :])
                if new_local_v >= old_local_v:
                    pi[s] = pi_t[s]
                else:
                    pi[s] = old_pi[s]
            else:
                pi[s] = pi_t[s]
        self.pi = pi
        if self.checks:
            self._check_if_lower_constrained_policy()

    def _check_if_lower_constrained_policy(self):
        lower_constrained = True
        for state in range(self.nb_states):
            distance = 0
            for action in range(self.nb_actions):
                distance += max(0, self.pi[state, action] - self.pi_b[state, action]) * self.errors[state, action]
            if distance > self.epsilon * (1 + 10 ** (-6)):
                lower_constrained = False
        if not lower_constrained:
            print(f'!!! The policy is not lower constrained !!!')


class AdvApproxSoftSPIBB(SoftSPIBB):
    # Implements Adv-Approx-Soft-SPIBB developed in 'Evaluation of Safe Policy Improvement by Soft Baseline
    # Bootstrapping' by Philipp Scholl.
    NAME = 'Adv-Approx-Soft-SPIBB'

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        super()._initial_calculations()
        if self.error_kind != 'mpeb':
            if self.speed_up_dict:
                self.q_pi_b_est = self.speed_up_dict['q_pi_b_est']
            else:
                self._compute_q_pi_b_samples()
                self._compute_q_pi_b_est_from_samples()

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """

        pi = np.zeros([self.nb_states, self.nb_actions])
        pi_t = self.pi_b.copy()
        for s in range(self.nb_states):
            budget_adv = 0
            A_bot = np.argsort(self.q[s, :])  # increasing order
            allowed_error = self.epsilon
            for a_bot in A_bot:
                mass_bot = min(pi_t[s, a_bot], allowed_error / (2 * self.errors[s, a_bot]))
                #  A_top is sorted in decreasing order:
                A_top = np.argsort(-(self.q[s, :] - self.q[s, a_bot]) / self.errors[s, :])

                for a_top in A_top:
                    if a_top == a_bot:
                        break

                    q_b_gain = self.q_pi_b_est[s, a_top] - self.q_pi_b_est[s, a_bot]
                    if q_b_gain < 0:
                        mass_top = min(mass_bot, allowed_error / (2 * self.errors[s, a_top]),
                                       budget_adv / (-1 * q_b_gain))
                    else:
                        mass_top = min(mass_bot, allowed_error / (2 * self.errors[s, a_top]))
                    if mass_top > 0:
                        budget_adv += q_b_gain * mass_top
                        mass_bot -= mass_top
                        pi_t[s, a_bot] -= mass_top
                        pi_t[s, a_top] += mass_top
                        allowed_error -= mass_top * (self.errors[s, a_bot] + self.errors[s, a_top])
                    if mass_bot == 0:
                        break
            # Local policy improvement check, required for convergence
            if self.old_pi is not None:
                new_local_v = pi_t[s, :].dot(self.q[s, :])
                old_local_v = self.old_pi[s, :].dot(self.q[s, :])
                if new_local_v >= old_local_v:
                    pi[s] = pi_t[s]
                else:
                    pi[s] = self.old_pi[s]
            else:
                pi[s] = pi_t[s]
        self.pi = pi
        if self.checks:
            self._check_if_constrained_policy()
            self._check_if_advantageous_policy()
        self.old_pi = self.pi.copy()
