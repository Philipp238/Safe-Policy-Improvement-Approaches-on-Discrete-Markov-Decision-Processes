# File to check manually the distribution of the performance of deterministic policies on the Wet Chicken and the
# Random MDPs benchmark.
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import configparser

# Set directory as the path to Evaluation-of-Safe-Policy-Improvement-with-Baseline-Bootstrapping
# directory = os.path.dirname(os.path.dirname(os.path.expanduser(__file__)))
directory = r'C:\Users\phili\PycharmProjects\Evaluation-of-Safe-Policy-Improvement-with-Baseline-Bootstrapping'

sys.path.append(directory)
path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'paths.ini'))
spibb_path = path_config['PATHS']['spibb_path']
sys.path.append(spibb_path)

import garnets
import spibb_utils
import spibb
import modelTransitions

from wet_chicken_discrete.dynamics import WetChicken


def sample_deterministic_policy(nb_states, nb_actions, bias=False):
    pi = np.zeros([nb_states, nb_actions])
    for state in range(nb_states):
        if ~ bias:
            pi[state, np.random.choice(nb_actions)] = 1
        else:
            # Just exploratory if there is a bias towards the first action
            x = 1 / 2 / (nb_actions - 1)
            p = [1 / 2]
            for i in range(nb_actions - 1):
                p.append(x)
            pi[state, np.random.choice(nb_actions, p=p)]
    return pi


seed = 1469231697
np.random.seed(seed)
log = True

### For Wet Chicken

length = 5
width = 5
max_turbulence = 3.5
max_velocity = 3

nb_states = length * width
nb_actions = 5

gamma = 0.95

wet_chicken = WetChicken(length=length, width=width, max_turbulence=max_turbulence,
                         max_velocity=max_velocity)

P = wet_chicken.get_transition_function()
R = wet_chicken.get_reward_function()
r_reshaped = spibb_utils.get_reward_model(P, R)

nb_samples = 1000000
list_unbiased_wet_chicken = []
for i in range(nb_samples):
    if i % 100000 == 0:
        print(f'{i} out of {nb_samples} done.')
    pi_sample = sample_deterministic_policy(nb_states, nb_actions, bias=False)
    list_unbiased_wet_chicken.append(spibb.policy_evaluation_exact(pi_sample, r_reshaped, P, gamma)[0][0])
df_unbiased_wet_chicken = pd.DataFrame(data=list_unbiased_wet_chicken, columns=['Performance'])

sns.set(font_scale=2)
g = sns.FacetGrid(data=df_unbiased_wet_chicken)
g.map(sns.ecdfplot, 'Performance')
g.fig.suptitle(f'ECDF of the performance of sampled deterministic policies on wet chicken')
plt.subplots_adjust(top=0.92, right=0.9, left=0.05, bottom=0.19)

### For RandomMDPs

gamma = 0.95
nb_states = 50
nb_actions = 4
nb_next_state_transition = 4
env_type = 2  # 1 for one terminal state, 2 for two terminal states

self_transitions = 0

garnet = garnets.Garnets(nb_states, nb_actions, nb_next_state_transition,
                         env_type=env_type, self_transitions=self_transitions)

# Only need this to correctly compute the garnet (setting the actual final state happens here...)
ratio = 0.5
softmax_target_perf_ratio = (ratio + 1) / 2
baseline_target_perf_ratio = ratio
garnet.generate_baseline_policy(gamma,
                                softmax_target_perf_ratio=softmax_target_perf_ratio,
                                baseline_target_perf_ratio=baseline_target_perf_ratio, log=False)

R = garnet.compute_reward()
P = garnet.transition_function

# Randomly pick a second terminal state and update model parameters
potential_final_states = [s for s in range(nb_states) if s != garnet.final_state and s != 0]
easter_egg = np.random.choice(potential_final_states)

assert (garnet.final_state != easter_egg)
R[:, easter_egg] = 1
P[easter_egg, :, :] = 0
r_reshaped = spibb_utils.get_reward_model(P, R)

nb_samples = 1000000
list_unbiased_random_mdps = []
pi_too_good = None
for i in range(nb_samples):
    if i % 100000 == 0:
        print(f'{i} out of {nb_samples} done.')
    pi_sample = sample_deterministic_policy(nb_states, nb_actions, bias=False)
    list_unbiased_random_mdps.append(spibb.policy_evaluation_exact(pi_sample, r_reshaped, P, gamma)[0][0])
    if spibb.policy_evaluation_exact(pi_sample, r_reshaped, P, gamma)[0][0] > 1.5:
        pi_too_good = pi_sample
df_unbiased_random_mdps = pd.DataFrame(data=list_unbiased_random_mdps, columns=['Performance'])
sns.set(font_scale=2)
g = sns.FacetGrid(data=df_unbiased_random_mdps)
g.map(sns.ecdfplot, 'Performance')
g.fig.suptitle(
    f'ECDF of the performance of sampled deterministic policies on Random MDPs with a good easter egg')
plt.subplots_adjust(top=0.92, right=0.9, left=0.05, bottom=0.19)

spibb.policy_evaluation_exact(pi_too_good, r_reshaped, P, gamma)[0]
