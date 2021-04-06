from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm

import numpy as np


class PiStar(BatchRLAlgorithm):
    # This class implements Dynamic Programming as described by 'Reinforcement Learning - An Introduction' by Sutton and
    # Barto by using the true reward matrix and transition probabilities. Even though it is not a Batch RL algorithm
    # this class inherits from BatchRLAlgorithm to be able to reuse its PE and PI step and also otherwise fit into
    # the framework, to make it easier to include the optimal policies in experiments.
    NAME = 'PI_STAR'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, P, episodic, checks=False, zero_unseen=True,
                 max_nb_it=5000, speed_up_dict=None):
        """
        As this class does not really implement a Batch RL algorithm, some of the input parameters can be set to None
        :param pi_b: not necessary, choice is not important
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: not necessary, choice is not important
        :param R: reward matrix as numpy array with shape (nb_states, nb_states), assuming that the reward is deterministic w.r.t. the
         previous and the next states
        :param P: true transition probabilities as numpy array with shape (nb_states, nb_actions, nb_states)
        :param episodic: boolean variable, indicating whether the MDP is episodic (True) or non-episodic (False)
        :param zero_unseen: not necessary, choice is not important
        :param max_nb_it: integer, indicating the maximal number of times the PE and PI step should be executed, if
        convergence is not reached
        :param checks: boolean variable indicating if different validity checks should be executed (True) or not
        (False); this should be set to True for development reasons, but it is time consuming for big experiments
        :param speed_up_dict: not necessary, choice is not important
        """
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
