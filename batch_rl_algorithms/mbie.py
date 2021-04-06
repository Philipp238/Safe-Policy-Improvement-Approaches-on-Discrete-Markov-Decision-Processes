import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class MBIE(BatchRLAlgorithm):
    # Pessimistic offline adaption of the optimistic online algorithm MBIE from 'An analysis of model-based Interval
    # Estimation for Markov Decision Processes' by Alexander L. Strehl and Michael L. Littman
    NAME = 'MBIE'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, delta, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None):
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
        algorithms, this should only be used for big experiments; for the standard algorithms this should only contain
        the following:
        'count_state_action': numpy array with shape (nb_states, nb_actions) indicating the number of times a s
        tate-action pair has been visited
        'count_state_action_state': numpy array with shape (nb_states, nb_actions, nb_states) indicating the number of
        times a state-action-next-state triplet has been visited
        """
        self.delta = delta
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen, max_nb_it, checks,
                         speed_up_dict)

    def _initial_calculations(self):
        '''
        Starts all the calculations which can be done before the actual training.
        '''
        super()._initial_calculations()
        self._compute_errors()

    def _compute_errors(self):
        '''
        Computes the error function relying on Hoeffding's bound.
        '''
        self.errors = np.zeros([self.nb_states, self.nb_actions])
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                self.errors[state, action] = np.sqrt(
                    2 * (np.log(2 * (self.nb_states * self.nb_actions) * (2 ** self.nb_states - 2) / self.delta)) /
                    self.count_state_action[state, action])
        self.errors[self.count_state_action == 0] = 2  # Maximal L1 distance between to prob. dists

    def _policy_evaluation(self):
        """
        Computes the worst case action-value function for the current policy self.pi.
        """
        q_pessimistic = self.q.copy()
        # q_pessimistic[self.mask_unseen] = 1 / (1 - self.gamma) * self.r_min
        nb_it = 0
        started = True
        while started or np.linalg.norm(q_pessimistic - old_q) > 10 ** (-9) and nb_it < self.max_nb_it:
            started = False
            nb_it += 1
            old_q = q_pessimistic.copy()
            V_pessimistic = np.max(old_q, axis=1)
            next_state_values = self.R_state_state[0] + self.gamma * V_pessimistic
            P_pessimistic = self.transition_model.copy()
            # Now find the worst P:
            for state in range(self.nb_states):
                for action in range(self.nb_actions):
                    epsilon = self.errors[state, action]
                    worst_next_state = np.argmin(next_state_values)
                    mass_added = np.min([epsilon / 2, 1 - P_pessimistic[state, action, int(worst_next_state)]])
                    P_pessimistic[state, action, worst_next_state] += mass_added
                    mass_subtracted = 0
                    V_top = np.argsort(-next_state_values)
                    for best_next_state in V_top:
                        if mass_added == mass_subtracted:
                            break
                        # Note: It can happen that the best_next_state is the same as the worst_next_state, i.e. if all
                        # states are equally good, then this just removes the probability mass and we end up not
                        # changing P_pessimistic at all, which is fine if every state's value is the same
                        mass_in_move = np.min(
                            [mass_added - mass_subtracted, P_pessimistic[state, action, best_next_state]])
                        P_pessimistic[state, action, best_next_state] -= mass_in_move
                        mass_subtracted += mass_in_move
            q_pessimistic = np.dot(P_pessimistic, next_state_values)
        self.q = q_pessimistic
