import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

import exploration_utils

config = configparser.ConfigParser()
config.read(os.path.join(r'C:\Users\phili\PycharmProjects\safe_rl', 'config.ini'))
path_results = config['PATHS']['results_path']

random_mdps = False
if random_mdps:
    directory = os.path.join(path_results, 'Random_mdps', 'Random_mdps_easter')
else:
    directory = os.path.join(path_results, 'wet_chicken', 'heuristic')
exp_path = os.path.join(directory, 'Experiments')
fig_path = os.path.join(directory, 'Figures')

if random_mdps:
    row = 'baseline_target_perf_ratio'
    col = 'nb_trajectories'
    y = 'normalized_perf'
else:
    row = 'epsilon_baseline'
    col = 'length_trajectory'
    y = 'method_perf'

results = pd.read_csv(os.path.join(exp_path, 'results_hyperparam_optimized.csv'))

if random_mdps:
    baseline_parameter_name = 'baseline_target_perf_ratio'
    baseline_parameter_name_displayed = 'baseline_perf_ratio'
    baseline_parameter = 0.9
    data_length_name = 'nb_trajectories'
else:
    baseline_parameter_name = 'epsilon_baseline'
    baseline_parameter_name_displayed = 'epsilon_baseline'
    results.drop(columns='baseline_method', inplace=True)
    baseline_parameter = 0.1
    data_length_name = 'length_trajectory'

results_of_interest = results[results[baseline_parameter_name] == baseline_parameter]

to_concatenate = []
methods = results_of_interest['method'].unique()
data_lengths = results_of_interest[data_length_name].unique()
for method in methods:
    print(f'Starting with {method} out of {methods}.')
    for data_length in data_lengths:
        print(f'Starting with date length {data_length} ot of {data_lengths}.')
        results_reduced = results_of_interest[
            (results['method'] == method) & (results[data_length_name] == data_length)]

        results_cvar = exploration_utils.get_cvar(results_of_interest=results_reduced, quantile=0.01, row=row, col=col,
                                                  random_mdp=random_mdps)
        np.random.choice(results_reduced.index, size=10000)

        df = pd.DataFrame({'x': [0, 1, 4, 9]})
        sample_choice = np.random.choice(df.index, size=4)
        sample = df.iloc[sample_choice]
        sample

        cvar_list = []
        nb_samples = 200
        sample_size = results_reduced.shape[0]
        for i in range(nb_samples):
            sample = results_reduced.loc[np.random.choice(results_reduced.index, size=sample_size)]
            sample_cvar = exploration_utils.get_cvar(results_of_interest=sample, quantile=0.01, row=row, col=col,
                                                     random_mdp=random_mdps)['method_perf']
            cvar_list.append(sample_cvar)

        cvar_std = np.sqrt(np.var(cvar_list))
        mean_std = np.sqrt(np.var(results_reduced['method_perf']) / sample_size)

        to_concatenate.append([method, baseline_parameter, data_length, mean_std, cvar_std])
        # var_list = []
        # sample_length_list = []
        # for i in range(10, 1000, 10):
        #     sample_length_list.append(i)
        #     var_list.append(np.var(cvar_list[:i]))

# np.var(results_reduced['method_perf']) / 10000
var_df = pd.DataFrame(columns=['method', 'baseline_parameter', 'data_length', 'mean_std', 'cvar_std'],
                           data=to_concatenate)
var_df.to_excel(os.path.join(exp_path, 'standard_derivation.xlsx'))