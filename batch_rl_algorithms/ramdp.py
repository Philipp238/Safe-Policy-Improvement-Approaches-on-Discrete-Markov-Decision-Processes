import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class RaMDP(BatchRLAlgorithm):
    # Algorithm from 'Safe policy improvement by minimizing robust baseline regret' by Marek Petrik, Yinlam Chow and
    # Mohammad Ghavamzadeh, which is also equivalent to MBIE-EB from 'An analysis of model-based Interval
    # Estimation for Markov Decision Processes' by Alexander L. Strehl and Michael L. Littman
    NAME = 'RaMDP'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, kappa, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None):
        self.kappa = kappa
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen, max_nb_it, checks,
                         speed_up_dict=speed_up_dict)

    def _compute_R_state_action(self):
        super()._compute_R_state_action()
        self.R_state_action -= self.kappa / np.sqrt(self.count_state_action)
        self.R_state_action[self.count_state_action == 0] = np.min(self.R_state_state) * (1 / (1 - self.gamma))
