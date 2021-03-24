import numpy as np


class BatchRLAlgorithm:
    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None):
        self.pi_b = pi_b
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.data = data  # The observed data of the behavior policy, should be a list of [s, a, s', r]
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
        self._build_model()
        self._compute_R_state_action()

    def fit(self):
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
        if self.episodic:
            batch_trajectory = [val for sublist in self.data for val in sublist]
        else:
            batch_trajectory = self.data.copy()
        self.count_state_action_state = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        for [action, state, next_state, _] in batch_trajectory:
            self.count_state_action_state[int(state), action, int(next_state)] += 1
        self.count_state_action = np.sum(self.count_state_action_state, 2)

    def _build_model(self):
        self.transition_model = self.count_state_action_state / self.count_state_action[:, :, np.newaxis]
        if self.zero_unseen:
            self.transition_model = np.nan_to_num(self.transition_model)
        else:
            self.transition_model[np.isnan(self.transition_model)] = 1. / self.nb_states

    def _compute_R_state_action(self):
        self.R_state_action = np.einsum('ijk,ik->ij', self.transition_model, self.R_state_state)

    def _policy_improvement(self):
        '''
        Updates the policy (Here: greedy update)
        :return:
        '''
        self.pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            self.pi[s, np.argmax(self.q[s, :])] = 1

    def _policy_evaluation(self):
        '''
        Computes q for self.pi
        :return:
        '''
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
