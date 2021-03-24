import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class SPIBB_abstract(BatchRLAlgorithm):
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, N_wedge, episodic, zero_unseen=True, checks=False,
                 max_nb_it=5000, speed_up_dict=None):
        self.N_wedge = N_wedge
        super().__init__(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=data, R=R,
                         zero_unseen=zero_unseen, max_nb_it=max_nb_it, episodic=episodic, checks=checks,
                         speed_up_dict=speed_up_dict)
        self.pi_b_masked = self.pi_b.copy()
        self.pi_b_masked[self.mask] = 0

    def _initial_calculations(self):
        super()._initial_calculations()
        self._compute_mask()

    def _compute_mask(self):
        self.mask = self.count_state_action > self.N_wedge


class SPIBB(SPIBB_abstract):
    NAME = 'SPIBB'

    def _policy_improvement(self):
        pi = self.pi_b_masked.copy()
        for s in range(self.nb_states):
            if len(self.q[s, self.mask[s]]) > 0:
                pi_b_masked_sum = np.sum(self.pi_b_masked[s])
                pi[s][np.where(self.mask[s])[0][np.argmax(self.q[s, self.mask[s]])]] = 1 - pi_b_masked_sum
        self.pi = pi


class Lower_SPIBB(SPIBB_abstract):
    NAME = 'Lower-SPIBB'

    def _policy_improvement(self):
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
