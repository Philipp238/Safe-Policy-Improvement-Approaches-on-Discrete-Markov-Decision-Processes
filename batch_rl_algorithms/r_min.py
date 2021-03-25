import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class RMin(BatchRLAlgorithm):
    # Pessimistic offline adaption of the optimistic online algorithm R-MAX from 'R-max - A General Polynomial Time
    # Algorithm for Near-Optimal Reinforcement Learning' from Ronen I. Brafman and Moshe Tennenholtz
    NAME = 'R_min'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, N_wedge, episodic, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None):
        self.N_wedge = N_wedge
        self.r_min = np.min(R)
        super().__init__(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=data, R=R,
                         zero_unseen=zero_unseen, max_nb_it=max_nb_it, episodic=episodic, checks=checks,
                         speed_up_dict=speed_up_dict)

    def _initial_calculations(self):
        super()._initial_calculations()
        self._compute_mask()

    def _policy_evaluation(self):
        '''
        Computes q as it is necessary for R_min: q[mask] = - V_max
        Could vary a bit from the old implementation R_min, as the convergence condition was changed. The reason for
        this is that it is not necessary to compute the exact q in every PE step and it did not seem to have an impact
        on the performance of the algorithm
        :return:
        '''
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
        self.mask = self.count_state_action > self.N_wedge
