import numpy as np

from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm


class MBIE(BatchRLAlgorithm):
    NAME = 'MBIE'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, episodic, delta, zero_unseen=True, max_nb_it=5000,
                 checks=False, speed_up_dict=None):
        self.delta = delta
        super().__init__(pi_b, gamma, nb_states, nb_actions, data, R, episodic, zero_unseen, max_nb_it, checks,
                         speed_up_dict)

    def _initial_calculations(self):
        super()._initial_calculations()
        self._compute_errors()

    def _compute_errors(self):
        self.errors = np.zeros([self.nb_states, self.nb_actions])
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                self.errors[state, action] = np.sqrt(
                    2 * (np.log(2 * (self.nb_states * self.nb_actions) * (2 ** self.nb_states - 2) / self.delta)) /
                    self.count_state_action[state, action])
        self.errors[self.count_state_action == 0] = 2  # Maximal L1 distance between to prob. dists

    def _policy_evaluation(self):
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
