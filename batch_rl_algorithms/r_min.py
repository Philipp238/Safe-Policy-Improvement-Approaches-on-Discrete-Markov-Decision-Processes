import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class RMin(BatchRLAlgorithm):
    # Pessimistic offline adaption of the optimistic online algorithm R-MAX from 'R-max - A General Polynomial Time
    # Algorithm for Near-Optimal Reinforcement Learning' from Ronen I. Brafman and Moshe Tennenholtz
    NAME = 'R_min'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, N_wedge, episodic, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None):
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
        :param N_wedge: Hyper-parameter of R-MIN
        """
        self.N_wedge = N_wedge
        self.r_min = np.min(R)
        super().__init__(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=data, R=R,
                         zero_unseen=zero_unseen, max_nb_it=max_nb_it, episodic=episodic, checks=checks,
                         speed_up_dict=speed_up_dict)

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        super()._initial_calculations()
        self._compute_mask()

    def _policy_evaluation(self):
        """
        Computes q as it is necessary for R_min: q[mask] = - V_max
        Could vary a bit from the old implementation R_min, as the convergence condition was changed. The reason for
        this is that it is not necessary to compute the exact q in every PE step and it did not seem to have an impact
        on the performance of the algorithm
        """
        self.q[~self.mask] = self.r_min * 1 / (1 - self.gamma)
        old_q = np.zeros([self.nb_states, self.nb_actions])
        nb_it = 0
        started = True
        while started or np.linalg.norm(self.q - old_q) > 0.00001 and nb_it < self.max_nb_it / 10:
            started = False
            nb_it += 1
            old_q = self.q.copy()
            for state in range(self.nb_states):
                for action in range(self.nb_actions):
                    if self.mask[state, action]:
                        # future_return = 0
                        # for next_state in range(self.nb_states):
                        #    future_return += self.P[state, action, next_state] * max(old_q[next_state])
                        # With the before code the algorithm needed 3.66 seconds for something RL took 0.0, SPIBB 0.008
                        # Now it takes 0.556 seconds...
                        future_return = np.dot(self.transition_model[state, action], np.max(old_q, axis=1))
                        self.q[state, action] = self.R_state_action[state, action] + self.gamma * future_return

    def _compute_mask(self):
        """
        Computes a boolean mask indicating whether a state-action pair has been more than N_wedge times.
        """
        self.mask = self.count_state_action > self.N_wedge
