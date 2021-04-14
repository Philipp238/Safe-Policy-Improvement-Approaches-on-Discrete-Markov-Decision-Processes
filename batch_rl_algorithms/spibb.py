import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class SPIBB_abstract(BatchRLAlgorithm):
    # Abstract base class for SPIBB (Pi_b-SPIBB) and Lower-SPIBB (Pi_<=b-SPIBB)
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, N_wedge, episodic, zero_unseen=True, checks=False,
                 max_nb_it=5000, speed_up_dict=None):
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
        algorithms, this should only be used for big experiments; for the standard algorithms this should only contain
        the following:
        'count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of times a s
        tate-action pair has been visited
        'count_state_action_state': numpy array with shape (nb_states, nb_actions, nb_states) indicating the number of
        times a state-action-next-state triplet has been visited
        :param N_wedge: hyper-parameter of SPIBB and Lower-SPIBB
        """
        self.N_wedge = N_wedge
        super().__init__(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=data, R=R,
                         zero_unseen=zero_unseen, max_nb_it=max_nb_it, episodic=episodic, checks=checks,
                         speed_up_dict=speed_up_dict)
        self.pi_b_masked = self.pi_b.copy()
        self.pi_b_masked[self.mask] = 0

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        super()._initial_calculations()
        self._compute_mask()

    def _compute_mask(self):
        """
        Computes a boolean mask indicating whether a state-action pair has been more than N_wedge times.
        """
        self.mask = self.count_state_action > self.N_wedge


class SPIBB(SPIBB_abstract):
    # Code and algorithm from 'Safe Policy Improvement with Baseline Bootstrapping' by Romain Laroche, Paul Trichelair
    # and Remi Tachet des Combes which implements Pi_b-SPIBB.
    # (Their code is available under: https://github.com/RomainLaroche/SPIBB)
    NAME = 'SPIBB'

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """
        pi = self.pi_b_masked.copy()
        for s in range(self.nb_states):
            if len(self.q[s, self.mask[s]]) > 0:
                pi_b_masked_sum = np.sum(self.pi_b_masked[s])
                pi[s][np.where(self.mask[s])[0][np.argmax(self.q[s, self.mask[s]])]] = 1 - pi_b_masked_sum
        self.pi = pi


class Lower_SPIBB(SPIBB_abstract):
    # Code and algorithm from 'Safe Policy Improvement with Baseline Bootstrapping' by Romain Laroche, Paul Trichelair
    # and Remi Tachet des Combeswhich implements Pi_<=b-SPIBB.
    #     # (Their code is available under: https://github.com/RomainLaroche/SPIBB)
    NAME = 'Lower-SPIBB'

    def _policy_improvement(self):
        """
        Updates the current policy self.pi.
        """
        self.pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            A = np.argsort(-self.q[s, :])
            pi_current_sum = 0
            for a in A:
                if self.mask[s, a] or self.pi_b[s, a] > 1 - pi_current_sum:
                    self.pi[s, a] = 1 - pi_current_sum
                    break
                else:
                    self.pi[s, a] = self.pi_b[s, a]
                    pi_current_sum += self.pi[s, a]
