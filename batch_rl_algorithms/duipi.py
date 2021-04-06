import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class DUIPI(BatchRLAlgorithm):
    # Algorithm implemented following 'Uncertainty Propagation for Efficient Exploration in Reinforcement Learning'
    # by Alexander Hans and Steffen Udluft; a small modification has been added see the Master's thesis
    # 'Evaluation of Safe Policy Improvement with Soft Baseline Bootstrapping'
    NAME = 'DUIPI'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, bayesian, xi, alpha_prior=0.1,
                 zero_unseen=True, max_nb_it=5000, checks=False, speed_up_dict=None):
        """
        :param pi_b: numpy matrix with shape (nb_states, nb_actions), such that pi_b(s,a) refers to the probability of
        choosing action a in state s by the behavior policy
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: the data collected by the behavior policy, which should be a list of [state, action, next_state,
         reward] sublists
        :param R: reward matrix as numpy array with shape (nb_states, nb_states), assuming that the reward is deterministic w.r.t. the
         previous and the next states
        :param episodic: boolean variable, indicating whether the MDP is episodic (True) or non-episodic (False)
        :param zero_unseen: boolean variable, indicating whether the estimated model should guess set all transition
        probabilities to zero for a state-action pair which has never been visited (True) or to 1/nb_states (False)
        :param max_nb_it: integer, indicating the maximal number of times the PE and PI step should be executed, if
        convergence is not reached
        :param checks: boolean variable indicating if different validity checks should be executed (True) or not
        (False); this should be set to True for development reasons, but it is time consuming for big experiments
        :param speed_up_dict: a dictionary containing pre-calculated quantities which can be reused by many different
        algorithms, this should only be used for big experiments; for DUIPI this should only contain
        the following:
        'count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of times a s
        tate-action pair has been visited
        'count_state_action_state': numpy array with shape (nb_states, nb_actions, nb_states) indicating the number of
        times a state-action-next-state triplet has been visited
        :param bayesian: boolean variable, indicating whether the estimation of the variance of the estimation of the
        transition probabilities should be done bayesian (True) using the Dirichlet distribution as a prior or
        frequentistic (False)
        :param xi: hyper-parameter of DUIPI, the higher xi is, the stronger is the influence of the variance
        :param alpha_prior: float variable necessary if bayesian=True, usually between 0 and 1
        """
        self.xi = xi
        self.alpha_prior = alpha_prior
        self.bayesian = bayesian
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen, max_nb_it, checks,
                         speed_up_dict)
        self.variance_q = np.zeros([self.nb_states, self.nb_actions])
        self.pi = 1 / self.nb_actions * np.ones([self.nb_states, self.nb_actions])

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        self._prepare_R_and_variance_R()
        self._prepare_P_and_variance_P()
        self._compute_mask()

    def _prepare_R_and_variance_R(self):
        """
        Estimates the reward matrix and its variance.
        """
        self.R_state_action_state = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        for action in range(self.nb_actions):
            self.R_state_action_state[:, action, :] = self.R_state_state.copy()
        self.variance_R = np.zeros([self.nb_states, self.nb_actions, self.nb_states])

    def _prepare_P_and_variance_P(self):
        """
        Estimates the reward matrix and its variance.
        """
        self.variance_P = np.zeros([self.nb_states, self.nb_actions, self.nb_states])
        if self.bayesian:
            alpha_d = (self.count_state_action_state + self.alpha_prior)
            alpha_d_0 = np.sum(alpha_d, 2)[:, :, np.newaxis]
            self.transition_model = alpha_d / alpha_d_0
            self.variance_P = alpha_d * (alpha_d_0 - alpha_d) / alpha_d_0 ** 2 / (alpha_d_0 + 1)
        else:
            self._build_model()
            for state in range(self.nb_states):
                self.variance_P[:, :, state] = self.transition_model[:, :, state] * (
                        1 - self.transition_model[:, :, state]) / (
                                                       self.count_state_action - 1)
            self.variance_P = np.nan_to_num(self.variance_P, nan=1 / 4,
                                            posinf=1 / 4)  # maximal variance is (b - a)^2 / 4
            self.variance_P[
                self.count_state_action == 0] = 1 / 4  # Otherwise variance_P would be if a state-action pair hasn't been visited yet
        self._check_if_valid_transitions()

    def _compute_mask(self):
        """
        Compute the mask which indicates which state-pair has never been visited.
        """
        self.mask = self.count_state_action > 0

    def _policy_evaluation(self):
        """
        Evaluates the current policy self.pi and calculates its variance.
        :return:
        """
        self.v = np.einsum('ij,ij->i', self.pi, self.q)
        self.variance_v = np.einsum('ij,ij->i', self.pi ** 2, self.variance_q)
        self.q = np.einsum('ijk,ijk->ij', self.transition_model, self.R_state_action_state + self.gamma * self.v)
        self.variance_q = np.dot(self.gamma ** 2 * self.transition_model ** 2, self.variance_v) + \
                          np.einsum('ijk,ijk->ij', (self.R_state_action_state + self.gamma * self.v) ** 2,
                                    self.variance_P) + \
                          np.einsum('ijk,ijk->ij', self.transition_model ** 2, self.variance_R)
        self.variance_q = np.nan_to_num(self.variance_q, nan=np.inf, posinf=np.inf)

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """
        q_uncertainty_and_mask_corrected = self.q - self.xi * np.sqrt(self.variance_q)
        # The extra modification to avoid unobserved state-action pairs
        q_uncertainty_and_mask_corrected[~self.mask] = - np.inf

        best_action = np.argmax(q_uncertainty_and_mask_corrected, axis=1)
        for state in range(self.nb_states):
            d_s = np.minimum(1 / self.nb_it, 1 - self.pi[state, best_action[state]])
            self.pi[state, best_action[state]] += d_s
            for action in range(self.nb_actions):
                if action == best_action[state]:
                    continue
                elif self.pi[state, best_action[state]] == 1:
                    self.pi[state, action] = 0
                else:
                    self.pi[state, action] = self.pi[state, action] * (1 - self.pi[state, best_action[state]]) / (
                            1 - self.pi[state, best_action[state]] + d_s)
