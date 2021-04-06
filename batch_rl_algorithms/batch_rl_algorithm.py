import numpy as np


class BatchRLAlgorithm:
    # Base class for all batch RL algorithms, which implements the general framework and the PE and PI step for
    # Dynamic Programming following 'Reinforcement Learning - An Introduction' by Sutton and Barto and
    # https://github.com/RomainLaroche/SPIBB. Additionally, it also implements the estimations of the transition
    # probabilities and reward matrix and some validation checks.
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen=True, max_nb_it=5000,
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
        self.pi_b = pi_b
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.data = data
        self.zero_unseen = zero_unseen
        self.episodic = episodic
        self.max_nb_it = max_nb_it
        self.pi = pi_b.copy()
        self.q = np.zeros([nb_states, nb_actions])
        self.R_state_state = R
        self.checks = checks
        self.speed_up_dict = speed_up_dict
        if self.speed_up_dict:
            self.count_state_action = self.speed_up_dict['count_state_action']
            self.count_state_action_state = self.speed_up_dict['count_state_action_state']
        else:
            self._count()
        self._initial_calculations()

    def _initial_calculations(self):
        """
        Starts all the calculations which can be done before the actual training.
        """
        self._build_model()
        self._compute_R_state_action()

    def fit(self):
        """
        Starts the actual training by reiterating between self._policy_evaluation() and self._policy_improvement()
        until convergence of the action-value function or the maximal number of iterations (self.max_nb_it) is reached.
        :return:
        """
        if self.checks:
            self._check_if_valid_transitions()
        old_q = np.ones([self.nb_states, self.nb_actions])
        self.nb_it = 0

        while np.linalg.norm(self.q - old_q) > 10 ** (-9) and self.nb_it < self.max_nb_it:
            self.nb_it += 1
            old_q = self.q.copy()
            self._policy_evaluation()
            self._policy_improvement()
            if self.checks:
                self._check_if_valid_policy()

        if self.nb_it > self.max_nb_it:
            with open("notconverging.txt", "a") as myfile:
                myfile.write(f"{self.NAME} is not converging. \n")

    def _count(self):
        """
        Counts the state-action pairs and state-action-triplets and stores them.
        """
        if self.episodic:
            batch_trajectory = [val for sublist in self.data for val in sublist]
        else:
            batch_trajectory = self.data.copy()
        self.count_state_action_state = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        for [action, state, next_state, _] in batch_trajectory:
            self.count_state_action_state[int(state), action, int(next_state)] += 1
        self.count_state_action = np.sum(self.count_state_action_state, 2)

    def _build_model(self):
        """
        Estimates the transition probabilities from the given data.
        """
        self.transition_model = self.count_state_action_state / self.count_state_action[:, :, np.newaxis]
        if self.zero_unseen:
            self.transition_model = np.nan_to_num(self.transition_model)
        else:
            self.transition_model[np.isnan(self.transition_model)] = 1. / self.nb_states

    def _compute_R_state_action(self):
        """
        Applies the estimated transition probabilities and the reward matrix with shape (nb_states, nb_states) to
        estimate a new reward matrix in the shape (nb_states, nb_actions) such that self.R_state_action[s, a] is the
        expected reward when choosing action a in state s in the estimated MDP.
        """
        self.R_state_action = np.einsum('ijk,ik->ij', self.transition_model, self.R_state_state)

    def _policy_improvement(self):
        """
        Updates the current policy self.pi (Here: greedy update).
        """
        self.pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            self.pi[s, np.argmax(self.q[s, :])] = 1

    def _policy_evaluation(self):
        """
        Computes the action-value function for the current policy self.pi.
        """
        nb_sa = self.nb_actions * self.nb_states
        M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.transition_model, self.pi).reshape(nb_sa, nb_sa)
        self.q = np.linalg.solve(M, self.R_state_action.reshape(nb_sa)).reshape(self.nb_states, self.nb_actions)

    def _check_if_valid_policy(self):
        checks = np.unique((np.sum(self.pi, axis=1)))
        valid = True
        for i in range(len(checks)):
            if np.abs(checks[i] - 0) > 10 ** (-6) and np.abs(checks[i] - 1) > 10 ** (-6):
                valid = False
        if not valid:
            print(f'!!! Policy not summing up to 1 !!!')

    def _check_if_valid_transitions(self):
        checks = np.unique((np.sum(self.transition_model, axis=2)))
        valid = True
        for i in range(len(checks)):
            if np.abs(checks[i] - 0) > 10 ** (-8) and np.abs(checks[i] - 1) > 10 ** (-8):
                valid = False
        if not valid:
            print(f'!!! Transitions not summing up to 0 or 1 !!!')

    def compute_safety(self):
        return {'Probability': None, 'lower_limit': None}

    @property
    def get_v(self):
        v = np.einsum('ij,ij->i', self.pi, self.q)
        return v

    # Get the advantage of the policy w.r.t. pi_b
    def get_advantage(self, state, q_pi_b_est):
        v_pi_b_est_state = q_pi_b_est[state] @ self.pi_b[state]
        advantage = (v_pi_b_est_state - q_pi_b_est[state]) @ self.pi[state]
        return advantage
