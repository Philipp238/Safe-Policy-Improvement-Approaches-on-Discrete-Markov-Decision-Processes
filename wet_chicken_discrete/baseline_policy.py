import numpy as np
import pandas as pd


class WetChickenBaselinePolicy:
    def __init__(self, env, gamma, method='heuristic', epsilon=0.1, convergence=0.1, learning_rate=0.1, max_nb_it=999,
                 order_epsilon=3, order_learning_rate=3):
        self.env = env
        self.gamma = gamma
        self.nb_states = env.width * env.length
        self.nb_actions = 5
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.epsilon = epsilon
        self.convergence = convergence
        self.learning_rate = learning_rate
        self.method = method
        self.max_nb_it = max_nb_it
        self.order_epsilon = order_epsilon
        self.order_learning_rate = order_learning_rate
        self.compute_baseline()

    def compute_baseline(self):
        if self.method == 'fixed_learning':
            old_q = np.zeros((self.nb_states, self.nb_actions))
            q = np.ones((self.nb_states, self.nb_actions)) * 1 / (1 - self.gamma) * 4  # Optimistic initialisation
            nb_it = 0
            state = self.env.get_state_int()
            # while (np.linalg.norm(old_q - q) > self.convergence) and nb_it < 999:
            while nb_it < self.max_nb_it:
                action = np.random.choice(self.pi.shape[1], p=self.pi[state])
                state, reward, next_state = self.env.step(action)
                old_q = q.copy()
                q[state, action] += self.learning_rate * (
                        reward + self.gamma * np.max(q[next_state, :]) - q[state, action])
                self.pi = self.epsilon * np.ones((self.nb_states, self.nb_actions)) / 5
                for s in range(self.nb_states):
                    self.pi[s, np.argmax(q[s, :])] += 1 - self.epsilon
                nb_it += 1
        elif self.method == 'variable_learning':
            old_q = np.zeros((self.nb_states, self.nb_actions))
            q = np.ones((self.nb_states, self.nb_actions)) * 1 / (1 - self.gamma) * 4  # Optimistic initialisation
            nb_it = 0
            state = self.env.get_state_int()
            # while (np.linalg.norm(old_q - q) > self.convergence) and nb_it < 999:
            while nb_it < self.max_nb_it:
                nb_it += 1
                epsilon = self.epsilon * 1 / nb_it ** (1 / self.order_epsilon)
                learning_rate = self.learning_rate * 1 / nb_it ** (1 / self.order_learning_rate)
                action = np.random.choice(self.pi.shape[1], p=self.pi[state])
                state, reward, next_state = self.env.step(action)
                old_q = q.copy()
                q[state, action] += learning_rate * (
                        reward + self.gamma * np.max(q[next_state, :]) - q[state, action])
                self.pi = epsilon * np.ones((self.nb_states, self.nb_actions)) / 5
                for s in range(self.nb_states):
                    self.pi[s, np.argmax(q[s, :])] += 1 - epsilon
        elif self.method == 'variable_learning':
            old_q = np.zeros((self.nb_states, self.nb_actions))
            q = np.ones((self.nb_states, self.nb_actions)) * 1 / (1 - self.gamma) * 4  # Optimistic initialisation
            nb_it = 0
            state = self.env.get_state_int()
            # while (np.linalg.norm(old_q - q) > self.convergence) and nb_it < 999:
            while nb_it < self.max_nb_it:
                nb_it += 1
                epsilon = self.epsilon * 1 / nb_it ** (1 / self.order_epsilon)
                learning_rate = self.learning_rate * 1 / nb_it ** (1 / self.order_learning_rate)
                action = np.random.choice(self.pi.shape[1], p=self.pi[state])
                state, reward, next_state = self.env.step(action)
                old_q = q.copy()
                q[state, action] += learning_rate * (
                        reward + self.gamma * np.max(q[next_state, :]) - q[state, action])
                self.pi = epsilon * np.ones((self.nb_states, self.nb_actions)) / 5
                for s in range(self.nb_states):
                    self.pi[s, np.argmax(q[s, :])] += 1 - epsilon
        elif self.method == 'state_count_dependent_variable':
            old_q = np.zeros((self.nb_states, self.nb_actions))
            q = np.ones((self.nb_states, self.nb_actions)) * 1 / (1 - self.gamma) * 4  # Optimistic initialisation
            nb_it = 0
            state = self.env.get_state_int()
            # while (np.linalg.norm(old_q - q) > self.convergence) and nb_it < 999:
            count_state_action = np.zeros((self.nb_states, self.nb_actions))
            while nb_it < self.max_nb_it:
                nb_it += 1
                epsilon = self.epsilon * 1 / nb_it ** (1 / self.order_epsilon)
                action = np.random.choice(self.pi.shape[1], p=self.pi[state])
                count_state_action[state, action] += 1
                learning_rate = self.learning_rate * 1 / count_state_action[state, action] ** (
                        1 / self.order_learning_rate)
                state, reward, next_state = self.env.step(action)
                old_q = q.copy()
                q[state, action] += learning_rate * (
                        reward + self.gamma * np.max(q[next_state, :]) - q[state, action])
                self.pi = epsilon * np.ones((self.nb_states, self.nb_actions)) / 5
                for s in range(self.nb_states):
                    self.pi[s, np.argmax(q[s, :])] += 1 - epsilon
        elif self.method == 'heuristic':
            # Try to get to in the middle of the river and then paddle as strong as possible against the stream
            # I.e. try to get to state (2,2), as a number 12, and then choose action 2
            pi = np.zeros((self.nb_states, self.nb_actions))
            for state in range(self.nb_states):
                for action in range(self.nb_actions):
                    x, y = int(state / self.nb_actions), state % self.nb_actions
                    if x > 2:
                        pi[state, 2] = 1  # We are too close to the waterfall ==> paddle as strong as possible
                    elif y < 2:
                        pi[state, 4] = 1  # We are not in immediate danger, but too close to the left ==> go right
                    elif y > 2:
                        pi[state, 3] = 1  # We are not in immediate danger, but too close to the right ==> go left
                    elif x == 2:
                        pi[state, 2] = 1  # We are perfect now, try to keep the position by paddling as strong as poss
                    elif x == 1:
                        pi[state, 1] = 1  # Close to perfect, just paddle a bit
                    else:
                        pi[state, 0] = 1  # Right lane, but too high up, just drift with the river
            self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi
        else:
            print(
                f'Method {self.method} is not available. Only acceptable methods are: \'heuristic\' and \'state_count_dependent_learning\' ')


class ContinuousWetChickenHeuristic:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def pi(self, state):
        x, y = state[0], state[1]
        pi = np.zeros(5)
        if x > 2.5:
            pi[2] = 1  # We are too close to the waterfall ==> paddle as strong as possible
        elif y < 2:
            pi[4] = 1  # We are not in immediate danger, but too close to the left ==> go right
        elif y > 3:
            pi[3] = 1  # We are not in immediate danger, but too close to the right ==> go left
        elif x > 2:
            pi[2] = 1  # We are perfect now, try to keep the position by paddling as strong as poss
        elif x > 1:
            pi[1] = 1  # Close to perfect, just paddle a bit
        else:
            pi[0] = 1  # Right lane, but too high up, just drift with the river
        pi = (1 - self.epsilon) * pi + self.epsilon * 1/5
        return pi
