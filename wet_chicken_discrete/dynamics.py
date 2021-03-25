import numpy as np

LENGTH = 5
WIDTH = 5
MAX_TURBULENCE = 3.5
MAX_VELOCITY = 3

ACTION_TRANSLATOR = {
    'Drift': np.zeros(2),
    'Neutral': np.array([-1, 0]),
    'Max': np.array([-2, 0]),
    'Left': np.array([0, -1]),
    'Right': np.array([0, 1])
}


class WetChicken:
    # Implements the 2-dimensional discrete Wet Chicken benchmark from 'Efficient Uncertainty Propagation for
    # Reinforcement Learning with Limited Data' by Alexander Hans and Steffen Udluft

    def __init__(self, length, width, max_turbulence, max_velocity, discrete=True):
        self.length = length
        self.width = width
        self.max_turbulence = max_turbulence
        self.max_velocity = max_velocity
        self._state = np.zeros(
            2)  # Don't use this state outside of this class, it is an array. Instead use get_state_int!
        self.discrete = discrete


    def get_state_int(self):
        return int(self._state[0] * self.width + self._state[1])

    def _get_reward(self):
        return self._state[0]

    def _velocity(self):
        return self.max_velocity * self._state[1] / self.width

    def _turbulence(self):
        return np.random.uniform(-1, 1) * (self.max_turbulence - self._velocity())

    def _reset(self):
        self._state = np.zeros(2)

    def step(self, action):
        # action_coordinates = ACTION_TRANSLATOR[action]
        action_coordinates = list(ACTION_TRANSLATOR.values())[action]
        x_hat = self._state[0] + action_coordinates[0] + self._velocity() + self._turbulence()
        y_hat = self._state[1] + action_coordinates[1]
        if self.discrete:
            old_state = self.get_state_int()
            x_hat = round(x_hat)
            y_hat = round(y_hat)
        else:
            old_state = self._state.copy()

        if x_hat >= self.length:
            self._state[0] = 0
        elif x_hat < 0:
            self._state[0] = 0
        else:
            self._state[0] = x_hat

        if x_hat >= self.length or y_hat < 0:
            self._state[1] = 0
        elif y_hat >= self.width:
            self._state[1] = self.width - 1
        else:
            self._state[1] = y_hat

        if self.discrete:
            new_state = self.get_state_int()
        else:
            new_state = self._state.copy()
        return old_state, self._get_reward(), new_state

    def _get_overlap(self, interval_1, interval_2):
        return max(0, min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]))

    def get_transition_function(self):
        if not self.discrete:
            raise AssertionError('You chose a continuous MDP, but requested the transition function.')
        nb_states = self.width * self.length
        nb_actions = len(ACTION_TRANSLATOR)
        P = np.zeros((nb_states, nb_actions, nb_states))
        for state in range(nb_states):
            x = int(state / self.length)
            y = state % self.width
            velocity = self.max_velocity * y / self.width
            turbulence = self.max_turbulence - velocity

            for action_nb, action in enumerate(ACTION_TRANSLATOR.keys()):
                action_coordinates = ACTION_TRANSLATOR[action]
                target_interval = [
                    x + action_coordinates[0] + velocity - turbulence,
                    x + action_coordinates[0] + velocity + turbulence]
                prob_mass_on_land = 1 / (2 * turbulence) * self._get_overlap(
                    [-self.max_turbulence - 2, -0.5],
                    target_interval)  # -self.max_turbulence - 2 should be the lowest possible
                prob_mass_waterfall = 1 / (2 * turbulence) * self._get_overlap(
                    [self.length - 0.5, self.length + self.max_turbulence + self.max_velocity],
                    target_interval)  # self.length + self.max_turbulence + self.max_velocity should be the highest possible
                y_hat = y + action_coordinates[1]
                if y_hat < 0:
                    y_new = 0
                elif y_hat >= self.width:
                    y_new = self.width - 1
                else:
                    y_new = y_hat
                y_new = int(y_new)
                P[state, action_nb, 0] += prob_mass_waterfall
                P[state, action_nb, y_new] += prob_mass_on_land
                for x_hat in range(self.width):
                    x_hat_interval = [x_hat - 0.5, x_hat + 0.5]
                    prob_mass = 1 / (2 * turbulence) * self._get_overlap(
                        x_hat_interval, target_interval)
                    P[state, action_nb, x_hat * self.length + y_new] += prob_mass
        return P

    def get_reward_function(self):
        if not self.discrete:
            raise AssertionError('You chose a continuous MDP, but requested the reward function.')
        nb_states = self.width * self.length
        R = np.zeros((nb_states, nb_states))
        for state in range(nb_states):
            for next_state in range(nb_states):
                R[state, next_state] = int(next_state / self.width)
        return R

    def evaluate_policy(self, nb_evaluations):
        performance = np.zeros(nb_evaluations)
        for i in nb_evaluations:
            continue
        ############### Do this in spibb and compare to the result if using the transition function


if __name__ == '__main__':
    wet_chicken = WetChicken(5, 5, 3.5, 3, discrete=False)
    action = 0
    print(wet_chicken.step(action))
    print(wet_chicken.step(action))
    print(wet_chicken.step(action))

