import torch
import numpy as np

from model_free_batch_rl_algorithms.model_free_batch_rl import BatchRLDQN

MAX_Q = 5


class AbstractCountingSubclasses(BatchRLDQN):

    def _initial_calculations(self):
        self.counts = {}
        for i in range(self.states.shape[0]):
            self._get_pseudo_count(self.states[i])

    def _get_pseudo_count(self, state):
        state_tuple = tuple(np.round(np.array(state, dtype=np.float), decimals=2))
        if not state_tuple in self.counts.keys():
            self.counts[state_tuple] = self.compute_pseudo_count(state)
        return self.counts[state_tuple]


class RaMDPDQN(AbstractCountingSubclasses):

    def __init__(self, data, network_size, pi_b, gamma, dim_state, nb_actions, episodic, learning_rate, kappa,
                 update_frequency=10, max_steps=1000, update_lr=100, batch_size=32, decay_rate=0.95):
        super().__init__(data=data, network_size=network_size, pi_b=pi_b, gamma=gamma, dim_state=dim_state,
                         nb_actions=nb_actions, episodic=episodic, learning_rate=learning_rate, update_lr=update_lr,
                         update_frequency=update_frequency, max_steps=max_steps, decay_rate=decay_rate)
        self.kappa = kappa

    def _get_bellman_target_batch(self, q_next_states, rewards, terminals, next_states, states, actions):
        basic_rl_bellman_target = super()._get_bellman_target_batch(q_next_states=q_next_states, rewards=rewards,
                                                                    terminals=terminals, next_states=next_states,
                                                                    states=states, actions=actions)
        reward_adjustment = BatchRLDQN.apply_to_slices(tensor=states, iterating_dimension=0,
                                                       func=self._get_pseudo_count)[torch.arange(self.batch_size),
                                                                                    actions]
        return basic_rl_bellman_target - self.kappa * reward_adjustment


class AbstractSPIBBDQN(AbstractCountingSubclasses):

    def _get_bellman_target_batch(self, q_next_states, rewards, terminals, next_states, states, actions):
        pi = BatchRLDQN.apply_to_slices(tensor=next_states, iterating_dimension=0, func=self.pi, unnormalized=False)
        return rewards + (1 - terminals) * self.gamma * torch.sum(pi * q_next_states, 1)


class SPIBBDQN(AbstractSPIBBDQN):

    def __init__(self, data, network_size, pi_b, gamma, dim_state, nb_actions, episodic, learning_rate, N_wedge,
                 update_frequency=10, max_steps=1000, update_lr=100, decay_rate=0.95):
        super().__init__(data=data, network_size=network_size, pi_b=pi_b, gamma=gamma, dim_state=dim_state,
                         nb_actions=nb_actions, episodic=episodic, learning_rate=learning_rate, update_lr=update_lr,
                         update_frequency=update_frequency, max_steps=max_steps, decay_rate=decay_rate)
        self.N_wedge = N_wedge

    def pi(self, state, unnormalized=True):
        if unnormalized:
            state = np.concatenate(self.scaler.transform(state.reshape(1, -1)))
        count = self._get_pseudo_count(state)
        # count = self.compute_pseudo_count(state)
        mask = (count >= self.N_wedge)
        pi_b = self.pi_b(state)
        state = torch.FloatTensor(state)
        q = self.network(state)
        actions = pi_b.copy()
        actions[mask] = 0

        actions_decreasing = torch.argsort(q, descending=True)
        for best_action in actions_decreasing:
            if mask[best_action]:
                actions[best_action] = sum(pi_b[mask])
                break
        return actions


class ApproxSoftSPIBBDQN(AbstractSPIBBDQN):

    def __init__(self, data, network_size, pi_b, gamma, dim_state, nb_actions, episodic, learning_rate, epsilon,
                 update_frequency=10, max_steps=1000, update_lr=100, batch_size=32, decay_rate=0.95):
        super().__init__(data, network_size, pi_b, gamma, dim_state, nb_actions, episodic, learning_rate,
                         update_frequency, max_steps, update_lr, batch_size, decay_rate)
        self.epsilon = epsilon

    def pi(self, state, unnormalized=True):
        if unnormalized:
            state = np.concatenate(self.scaler.transform(state.reshape(1, -1)))
        counts = torch.FloatTensor(self._get_pseudo_count(state))
        errors = torch.sqrt(1 / (counts + 1e-9))
        # pi_b = torch.FloatTensor(self.pi_b(state))
        pi_b = self.pi_b(state)
        state = torch.FloatTensor(state)
        q = self.network(state)
        # actions = pi_b.clone().detach()
        actions = pi_b.copy()
        allowed_error = self.epsilon
        sorted_qs, arg_sorted_qs = torch.sort(q)
        # dp = torch.arange(self.minibatch_size)
        # sorted_pi_b = pi_b[dp[:, None], arg_sorted_qs]
        # sorted_errors = errors[dp[:, None], arg_sorted_qs]

        for a_bot in arg_sorted_qs:
            mass_bot = torch.min(pi_b[a_bot], allowed_error / (2 * errors[a_bot]))
            A_top = torch.argmax(q - q[a_bot] / errors)
            mass_top = torch.min(mass_bot, allowed_error / (2 * errors[A_top]))
            mass_bot -= mass_top
            actions[a_bot] -= mass_top
            actions[A_top] += mass_top
            allowed_error -= mass_top * (errors[a_bot] + errors[A_top])

        normalizing_constant = np.sum(actions)
        if np.abs(normalizing_constant - 1) > 0.01:
            raise ValueError('Pi is not a policy (probabilities do not sum up to 1).')
        return actions/normalizing_constant

    def batchwise(self, q_next_states, rewards, terminals, next_states, states, actions):
        counts = torch.FloatTensor(self._get_pseudo_count(states))
        errors = torch.sqrt(1 / (counts + 1e-9))
        pi_b = self.pi_b(states)
        state = torch.FloatTensor(states)
        q = self.network(state)
        actions = pi_b.copy()
        allowed_error = self.epsilon * torch.ones((self.batch_size))
        sorted_qs, arg_sorted_qs = torch.sort(q, dim=1)
        dp = torch.arange(self.batch_size)
        sorted_pi_b = pi_b[dp[:, None], arg_sorted_qs]
        sorted_errors = errors[dp[:, None], arg_sorted_qs]
        for a_bot in range(self.nb_actions):
            mass_bot = torch.min(sorted_pi_b[:, a_bot], allowed_error / (2 * sorted_errors[:, a_bot]))
            _, A_top = torch.max((q - sorted_qs[:, a_bot][:, None]) / errors, dim=1)
            mass_top = torch.min(mass_bot, allowed_error / (2 * errors[dp, A_top]))
            mass_bot -= mass_top
            actions[dp, arg_sorted_qs[:, a_bot]] -= mass_top
            actions[dp, A_top] += mass_top
            allowed_error -= mass_top * (sorted_errors[:, a_bot] + errors[dp, A_top])
        return rewards + (1 - terminals) * self.gamma * torch.sum(actions * q_next_states, 1)
