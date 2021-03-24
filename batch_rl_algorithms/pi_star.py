from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm

import numpy as np


class PiStar(BatchRLAlgorithm):
    NAME = 'PI_STAR'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, P, episodic, checks=False, zero_unseen=True,
                 max_nb_it=5000, speed_up_dict=None):
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.zero_unseen = zero_unseen
        self.episodic = episodic
        self.max_nb_it = max_nb_it
        self.pi = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions
        self.q = np.zeros([nb_states, nb_actions])
        self.R_state_state = R
        self.checks = checks
        self.transition_model = P
        self._compute_R_state_action()
