import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

from model import SmallDenseNetwork, DenseNetwork, Network, LargeNetwork, NatureNetwork


class BatchRLDQN:
    def __init__(self, data, network_size, pi_b, gamma, dim_state, nb_actions, episodic, learning_rate,
                 update_frequency=10, max_steps=1000, update_lr=100, batch_size=32, decay_rate=0.95):
        self.scaler = StandardScaler()
        self.dim_state = dim_state
        self.nb_actions = nb_actions
        self.data = data  # batch_trajectory out of state, action, next_state, reward, terminal
        self.episodic = episodic
        self._transform_data()
        self.gamma = gamma
        self.network_size = network_size
        self.pi_b = pi_b
        self.device = 'cpu'
        self.network = self._build_network()
        self.target_network = self._build_network()
        self._weight_transfer(from_model=self.network, to_model=self.target_network)
        self.max_steps = max_steps
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.update_lr = update_lr  # not used currently
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=1e-07)
        self._initial_calculations()
        self.batch_size = batch_size
        self.decay_rate = decay_rate

        # For debugging reasons
        self.list_of_loss = []

    def _transform_data(self):
        # constructs batch data out of self.data and adds the terminal variable
        self.unnormalized_states = np.zeros((len(self.data), self.dim_state))
        self.actions = np.zeros(len(self.data), dtype=np.int)
        self.rewards = np.zeros(len(self.data))
        self.unnormalized_next_states = np.zeros((len(self.data), self.dim_state))
        self.terminals = np.zeros(len(self.data))
        if self.episodic:
            raise NotImplementedError('Episodic data transformation is not implemented yet.')
            # batch_trajectory = [val + [i == len(sublist) - 1] for sublist in self.data for i, val in enumerate(sublist)]
        else:
            for i, observation in enumerate(self.data):
                self.unnormalized_states[i], self.actions[i], self.rewards[i], self.unnormalized_next_states[
                    i] = observation
        self.scaler.fit(self.unnormalized_states)
        self.states = self.scaler.transform(self.unnormalized_states)
        self.next_states = self.scaler.transform(self.unnormalized_next_states)

    def _build_network(self):
        if self.network_size == 'small':
            return Network()
        elif self.network_size == 'large':
            return LargeNetwork(state_shape=self.dim_state, nb_channels=4, nb_actions=self.nb_actions,
                                device=self.device)
        elif self.network_size == 'nature':
            return NatureNetwork(state_shape=self.dim_state, nb_channels=4, nb_actions=self.nb_actions,
                                 device=self.device)
        elif self.network_size == 'dense':
            return DenseNetwork(state_shape=self.dim_state, nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'small_dense':
            return SmallDenseNetwork(state_shape=self.dim_state, nb_actions=self.nb_actions, device=self.device)
        else:
            raise ValueError('Invalid network_size.')

    @staticmethod
    def _weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def fit(self):
        self.step = 0
        while self.step < self.max_steps:
            self.step += 1
            if (self.step % self.update_frequency) == 0:
                self._weight_transfer(from_model=self.network, to_model=self.target_network)
            states, actions, rewards, next_states, terminals = self.sample_experience_replay()
            self._update_batch(states, actions, rewards, next_states, terminals)
            if (self.step % (int(len(self.data) / self.batch_size))) == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= self.decay_rate

    def _update_batch(self, states, actions, rewards, next_states, terminals):
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        terminals = torch.FloatTensor(np.float32(terminals)).to(self.device)

        # Squeeze dimensions for history_len = 1
        states = torch.squeeze(states)
        next_states = torch.squeeze(next_states)
        q_preds = self.network(states)
        q_preds = q_preds[torch.arange(self.batch_size), actions]
        # q_pred = q_pred.gather(1, action.unsqueeze(1)).squeeze(1)
        q_next_states = self.target_network(next_states).detach()

        bellman_target = self._get_bellman_target_batch(q_next_states=q_next_states, rewards=rewards,
                                                        terminals=terminals, next_states=next_states,
                                                        states=states, actions=actions)

        # Huber loss
        errs = (bellman_target - q_preds)
        quad = torch.min(torch.abs(errs), torch.ones(len(errs)))
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for debugging reasons
        self.list_of_loss.append(loss.cpu().detach().numpy())
        return loss

    def _get_bellman_target_batch(self, q_next_states, rewards, terminals, next_states, states, actions):
        q_target_max, _ = torch.max(q_next_states, dim=1)
        return rewards + (1 - terminals) * self.gamma * q_target_max.detach()

    def sample_experience_replay(self):
        indices = np.random.random_integers(0, len(self.data) - 1, self.batch_size)
        # states, actions, rewards, next_states, terminals = [], [], [], [], []
        # for index in indices:
        #     state, action, reward, next_state, terminal = self.data[index]
        #     states.append(state)
        #     actions.append(action)
        #     rewards.append(reward)
        #     next_states.append(next_state)
        #     terminals.append(terminal)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], \
               self.terminals[indices]

    def compute_pseudo_count(self, state):
        pseudo_count = np.zeros(self.nb_actions)
        # for [current_state, action, reward, next_state, terminal] in self.data:
        #     pseudo_count[action] += max(0, 1 - np.linalg.norm(state - current_state, ord=2))
        for action in range(self.nb_actions):
            mask = self.actions == action
            pseudo_count[action] = np.sum(np.maximum(
                np.zeros(np.sum(mask)), 1 - np.linalg.norm(state - self.states[mask], ord=2, axis=1)))
        return pseudo_count

    def pi(self, state, unnormalized=True):
        if unnormalized:
            state = np.concatenate(self.scaler.transform(state.reshape(1, -1)))
        actions = torch.zeros(self.nb_actions)
        state = torch.FloatTensor(state)
        q_target = self.target_network(state).detach()
        actions[torch.argmax(q_target)] = 1
        return actions

    def _initial_calculations(self):
        pass

    @staticmethod
    def apply_to_slices(tensor, iterating_dimension, func, **kwargs):
        new_tensor = []
        for i in range(tensor.shape[iterating_dimension]):
            new_tensor.append(func(tensor[i], **kwargs))
        return torch.FloatTensor(new_tensor)
