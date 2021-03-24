import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

from exploration_utils import get_results_hyperparam_optimized
from exploration_utils import runtime_ecdf, load_data_first_time

config = configparser.ConfigParser()
config.read(os.path.join(r'C:\Users\phili\PycharmProjects\safe_rl', 'config.ini'))
path_results = config['PATHS']['results_path']

### Prepare own experiments ###
exp_path = os.path.join(path_results, 'Random_mdps', 'Random_mdps_easter', 'Experiments', 'mpeb_and_new_duipi')
results_mpeb_duipi = load_data_first_time(exp_path)
results_mpeb_duipi['normalized_perf'] = (results_mpeb_duipi['method_perf'] - results_mpeb_duipi['baseline_perf']) / (
        results_mpeb_duipi['pi_star_perf'] - results_mpeb_duipi['baseline_perf'])
results_duipi = results_mpeb_duipi[results_mpeb_duipi['method'] == 'DUIPI_bayesian']
results_mpeb = results_mpeb_duipi[results_mpeb_duipi['method'] != 'DUIPI_bayesian']

results_mpeb.drop(columns='Unnamed: 0', inplace=True)
results_mpeb.to_csv(os.path.join(exp_path, 'results_mpeb_full_normalized.csv'))

hyperparameters_soft_spibb = {'1-Step-Approx-Soft-SPIBB_hoeffding': 1,
                              'Approx-Soft-SPIBB_hoeffding': 2,
                              'Adv-Approx-Soft-SPIBB_hoeffding': 2,
                              'Lower-Approx-Soft-SPIBB_hoeffding': 1,
                              '1-Step-Approx-Soft-SPIBB_mpeb': 2,
                              'Approx-Soft-SPIBB_mpeb': 5,
                              'Adv-Approx-Soft-SPIBB_mpeb': 5,
                              'Lower-Approx-Soft-SPIBB_mpeb': 5}
results_hyperparam_optimized_soft_spibb = get_results_hyperparam_optimized(results_mpeb, hyperparameters_soft_spibb)
results_hyperparam_optimized_soft_spibb.to_csv(
    os.path.join(exp_path, 'soft_spibb_mpeb_hoeffding_hyperparam_optimized.csv'))

results.columns
cols = [1, 2, 3, 4]
results.drop(results.columns[cols], axis=1, inplace=True)
results.to_csv(
    os.path.join(path_results, 'Random_mdps_normal', 'Experiments', 'results_reduced_normalized_retyped.csv'))
results['method'].astype('category')

### Load normal random mdps ###
exp_path = os.path.join(path_results, 'Random_mdps', 'Random_mdps_easter', 'Experiments')
results = pd.read_csv(os.path.join(exp_path, 'results_reduced_normalized.csv'), index_col=False)

results_old_duipi_bayesian = results[results['method'] == 'DUIPI_bayesian']
results_not_duipi = results[results['method'].isin(['DUIPI_bayesian', 'DUIPI_frequentist'])]
cols = [0, 2, 3, 4, 5]
results_duipi.drop(results_duipi.columns[cols], axis=1, inplace=True)
results = pd.concat([results_not_duipi, results_duipi])
results.to_csv(os.path.join(exp_path, 'results_reduced_normalized.csv'))
results_duipi.to_csv(os.path.join(exp_path, 'results_duipi_new_reduced_normalized.csv'))

results_duipi_hyperparam_optimized = results_duipi[results_duipi['hyperparam'] == 0.1]
results_hyperparam_optimized = pd.concat([results_duipi_hyperparam_optimized, results_hyperparam_optimized_not_duipi])

results_hyperparam_optimized_not_duipi = results_hyperparam_optimized_not_duipi[
    ~ results_hyperparam_optimized_not_duipi['method'].isin(['DUIPI_bayesian', 'DUIPI_frequentist'])]

results_hyperparam_optimized_not_duipi['method'].unique()

results_hyperparam_optimized.to_csv(os.path.join(exp_path, 'results_hyperparam_optimized.csv'))

results_duipi['method'].unique()
results_old_duipi_bayesian['method'].unique()
results_old_duipi_bayesian['method'] = 'DUIPI_bayesian_native'
results_old_and_new_duipi_bayesian = pd.concat([results_duipi, results_old_duipi_bayesian])
results_old_and_new_duipi_bayesian.to_csv(os.path.join(exp_path, 'results_duipi_old_and_new.csv'))

results_old_and_new_duipi_bayesian_hyperparam_optimized = get_results_hyperparam_optimized(
    results_old_and_new_duipi_bayesian, hyperparameters)

results_old_and_new_duipi_bayesian_hyperparam_optimized.to_csv(
    os.path.join(exp_path, 'results_duipi_old_and_new_hyperparam_optimized.csv'))
# int_cols = ['iteration', 'nb_trajectories']
# float_cols = ['softmax_target_perf_ratio', 'baseline_target_perf_ratio', 'baseline_perf', 'pi_rand_perf',
#               'pi_star_perf', 'hyperparam', 'method_perf', 'time', 'normalized_perf']
# category_cols = ['seed', 'nb_trajectories', 'softmax_target_perf_ratio', 'baseline_target_perf_ratio', 'method']
# results[int_cols] = results[int_cols].astype('int32')
# results[float_cols] = results[float_cols].astype('float32')
# results[category_cols] = results[category_cols].astype('category')
# results.memory_usage()
# results.info()
# results_int = results.select_dtypes(include=['int'])
# results_int_converted = results_int.apply(pd.to_numeric, downcast='unsigned')
# results[results_int_converted.columns] = results_int_converted
# results.info()
# results['baseline_target_perf_ratio'] = results['baseline_target_perf_ratio'].round(decimals=1)

### Load easter egg first time ###
exp_path = os.path.join(path_results, 'Random_mdps', 'Random_mdps_easter', 'Experiments')
dirs = os.listdir(exp_path)
results_list = []
for dir in dirs:
    if dir.endswith('xlsx'):
        continue
    files = os.listdir(os.path.join(exp_path, dir))
    for file in files:
        if file.endswith('.csv'):
            print(os.path.join(exp_path, dir, file))
            results_list.append(pd.read_csv(os.path.join(exp_path, dir, file)))

results = pd.concat(results_list)
results['normalized_perf'] = (results['method_perf'] - results['baseline_perf']) / (
        results['pi_star_perf'] - results['baseline_perf'])
results.drop(columns='Unnamed: 0', inplace=True)
results.to_csv(os.path.join(exp_path, 'results_full_normalized.csv'), index=False)
cols = [1, 2, 3, 4]
results.drop(results.columns[cols], axis=1, inplace=True)
results.to_csv(os.path.join(exp_path, 'results_reduced_normalized.csv'), index=False)

# Load easter egg second time
exp_path = os.path.join(path_results, 'Random_mdps', 'Random_mdps_easter', 'Experiments')
results = pd.read_csv(os.path.join(exp_path, 'results_full_normalized.csv'))

hyperparameters_soft_spibb = {'1-Step-Approx-Soft-SPIBB_hoeffding': 1,
                              'Approx-Soft-SPIBB_hoeffding': 2,
                              'Adv-Approx-Soft-SPIBB_hoeffding': 2,
                              'Lower-Approx-Soft-SPIBB_hoeffding': 1}
hyperparameters = {**hyperparameters_soft_spibb,
                   'DUIPI_bayesian_native': 0.5,
                   'DUIPI_bayesian': 0.1,
                   'DUIPI_frequentist': 0.1,
                   'R_min': 3,
                   'SPIBB': 10,
                   'Lower-SPIBB': 10,
                   'RaMDP': 0.05
                   }

results_hyperparam_optimized_soft_spibb = get_results_hyperparam_optimized(results, hyperparameters_soft_spibb)
results_hyperparam_optimized = get_results_hyperparam_optimized(results, hyperparameters)

results_hyperparam_optimized_soft_spibb.to_csv(os.path.join(exp_path, 'results_hyperparam_optimized_soft_spibb.csv'),
                                               index=False)
results_hyperparam_optimized.to_csv(os.path.join(exp_path, 'results_hyperparam_optimized.csv'), index=False)

results_hyperparam_optimized_soft_spibb['method'].head()
results['method'].unique()
results_hyperparam_optimized['method'].unique()

results_hyperparam_optimized_soft_spibb['method'].unique()

sum(results['method'] == 'Adv-Approx-Soft-SPIBB_hoeffding')

### for the second time ###
results_hyperparam_optimized_all = pd.concat([results_hyperparam_optimized, results[
    results['method'].isin(['Basic_rl', 'R_min', 'MBIE', 'RaMDP', 'Pi_b_spibb', 'Pi_<b_SPIBB'])]])
methods_to_skip = ['Adv-Approx-Soft_SPIBB-e_min', 'Approx-Soft-SPIBB-e_min', 'MBIE']

### Load wet_chicken heuristic ###

exp_path = os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments')
dirs = os.listdir(exp_path)
results_list = []
for dir in dirs:
    if os.path.isdir(os.path.join(exp_path, dir)):
        print(os.path.join(exp_path, dir))
        files = os.listdir(os.path.join(exp_path, dir))
        for file in files:
            if file.endswith('.csv'):
                print(file)
                results_list.append(pd.read_csv(os.path.join(exp_path, dir, file)))

results = pd.concat(results_list)

exp_path = os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments', 'mpeb')
dirs = os.listdir(exp_path)
results_mpeb_list = []
for dir in dirs:
    if os.path.isdir(os.path.join(exp_path, dir)):
        print(os.path.join(exp_path, dir))
        files = os.listdir(os.path.join(exp_path, dir))
        for file in files:
            if file.endswith('.csv'):
                print(file)
                results_mpeb_list.append(pd.read_csv(os.path.join(exp_path, dir, file)))

results_mpeb = pd.concat(results_mpeb_list)
results_mpeb['normalized_perf'] = (results_mpeb['method_perf'] - results_mpeb['pi_b_perf']) / (
        results_mpeb['pi_star_perf'] - results_mpeb['pi_b_perf'])
results_mpeb.drop(columns='Unnamed: 0', inplace=True)
results_mpeb.to_csv(os.path.join(exp_path, 'results_reduced_normalized_soft_spibb.csv'))
len(results_mpeb)
mpeb_methods = [method for method in results_mpeb['method'].unique() if method.endswith('mpeb')]

len(results_mpeb[results_mpeb['method'].isin(mpeb_methods)])
results_only_mpeb = results_mpeb[results_mpeb['method'].isin(mpeb_methods)]

results_not_mpeb = pd.read_csv(
    os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments', 'results_full_normalized.csv'))
results = pd.concat([results_not_mpeb] + [results_only_mpeb])

results.to_csv(os.path.join(exp_path, 'results_full_normalized.csv'), index=False)
cols = [1, 2, 3, 4, 5]
results.drop(results.columns[cols], axis=1, inplace=True)
results.to_csv(os.path.join(exp_path, 'results_reduced_normalized.csv'))

results = pd.read_csv(
    os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments', 'results_reduced_normalized.csv'))

# With DUIPI & MBIE
exp_path = os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments_DUIPI_MBIE')
dirs = os.listdir(exp_path)
results_list = []
for dir in dirs:
    if os.path.isdir(os.path.join(exp_path, dir)):
        print(os.path.join(exp_path, dir))
        files = os.listdir(os.path.join(exp_path, dir))
        for file in files:
            if file.endswith('.csv'):
                print(file)
                results_list.append(pd.read_csv(os.path.join(exp_path, dir, file)))

results_1 = pd.concat(results_list)
results_1['normalized_perf'] = (results_1['method_perf'] - results_1['pi_b_perf']) / (
        results_1['pi_star_perf'] - results_1['pi_b_perf'])
results_1.drop(columns='Unnamed: 0', inplace=True)
results_1.to_csv(os.path.join(exp_path, 'results_full_normalized.csv'), index=False)
cols = [1, 2, 3, 4, 5]
results_1.drop(results_1.columns[cols], axis=1, inplace=True)
results_1.to_csv(os.path.join(exp_path, 'results_duipi_mbie_reduced_normalized.csv'))

# All
results_all = pd.concat([results_without_duipi_mbie, results_duipi_mbie])
results_all.to_csv(
    os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments', 'results_reduced_normalised.csv'))

### Load wet_chicken heuristic a second time ###
exp_path = os.path.join(path_results, 'wet_chicken', 'heuristic', 'Experiments')
results = pd.read_csv(os.path.join(exp_path, 'results_reduced_normalized.csv'))
# Old
# hyperparameters_soft_spibb = {'1-Step-Exact-Soft-SPIBB-Hoeffding': 0.1,
#                               '1-Step-Approx-Soft-SPIBB-Hoeffding': 1,
#                               'Exact-Soft-SPIBB-Hoeffding': 1,
#                               'Approx-Soft-SPIBB-Hoeffding': 1,
#                               'Approx-Soft-SPIBB-e_min': 1,
#                               'Adv-Approx-Soft_SPIBB-Hoeffding': 1,
#                               'Adv-Approx-Soft_SPIBB-e_min': 1}
# hyperparameters = {**hyperparameters_soft_spibb,
#                    'DUIPI_frequentist': 0.1,
#                    'DUIPI_bayesian': 0.1,
#                    'Pi_<b_SPIBB': 7,
#                    'Pi_b_spibb': 7,
#                    'R_min': 7,
#                    'RaMDP': 0.05
#                    }
hyperparameters_soft_spibb = {'1-Step-Approx-Soft-SPIBB_hoeffding': 0.2,
                              'Approx-Soft-SPIBB_hoeffding': 1,
                              'Adv-Approx-Soft-SPIBB_hoeffding': 1,
                              'Lower-Approx-Soft-SPIBB_hoeffding': 0.5,
                              '1-Step-Approx-Soft-SPIBB_mpeb': 0.5,
                              'Approx-Soft-SPIBB_mpeb': 5,
                              'Adv-Approx-Soft-SPIBB_mpeb': 5,
                              'Lower-Approx-Soft-SPIBB_mpeb': 5}
hyperparameters = {**hyperparameters_soft_spibb,
                   'DUIPI_frequentist': 0.05,
                   'DUIPI_bayesian': 0.5,
                   'Lower-SPIBB': 7,
                   'SPIBB': 7,
                   'R_min': 3,
                   'RaMDP': 2
                   }
results_hyperparam_optimized_soft_spibb = get_results_hyperparam_optimized(results, hyperparameters_soft_spibb)
results_hyperparam_optimized = get_results_hyperparam_optimized(results, hyperparameters)
results_hyperparam_optimized_soft_spibb.to_csv(os.path.join(exp_path, 'results_hyperparam_optimized_soft_spibb.csv'),
                                               index=False)
results_hyperparam_optimized.to_csv(os.path.join(exp_path, 'results_hyperparam_optimized.csv'), index=False)

sum(results['method'] == '1-Step-Approx-Soft-SPIBB_Hoeffding')
results['method'].unique()
### Run time analysis ###
path = os.path.join(path_results, r'Random_mdps_easter\Figures\Runtime\ECDF')
runtime_ecdf(results, path)

### Load bad easter egg first time ###
exp_path = os.path.join(path_results, 'Random_mdps_malicious', 'Experiments')
dirs = os.listdir(exp_path)
results_list = []
for dir in dirs:
    if dir.startswith('bad'):
        files = os.listdir(os.path.join(exp_path, dir))
        for file in files:
            if file.endswith('.xlsx'):
                print(file)
                results_list.append(pd.read_excel(os.path.join(exp_path, dir, file)))

results = pd.concat(results_list)
results['normalized_perf'] = (results['method_perf'] - results['baseline_perf']) / (
        results['pi_star_perf'] - results['baseline_perf'])
results.drop(columns='Unnamed: 0', inplace=True)
results.to_csv(os.path.join(exp_path, 'results_full_normalized.csv'), index=False)
cols = [1, 2, 3, 4]
results.drop(results.columns[cols], axis=1, inplace=True)
results.to_csv(os.path.join(exp_path, 'results_reduced_normalized.csv'))

# Load bad easter egg second time
results = pd.read_csv(os.path.join(exp_path, 'results_full_normalized.csv'))

hyperparameters_soft_spibb = {'1-Step-Exact-Soft-SPIBB-Hoeffding': 0.2,
                              '1-Step-Approx-Soft-SPIBB-Hoeffding': 0.5,
                              'Exact-Soft-SPIBB-Hoeffding': 0.5,
                              'Approx-Soft-SPIBB-Hoeffding': 1,
                              'Approx-Soft-SPIBB-e_min': 1,
                              'Adv-Approx-Soft_SPIBB-Hoeffding': 1,
                              'Adv-Approx-Soft_SPIBB-e_min': 1}
hyperparameters = {**hyperparameters_soft_spibb,
                   'DUIPI': 0.1}
results_hyperparam_optimized_soft_spibb = get_results_hyperparam_optimized(results, hyperparameters_soft_spibb)
results_hyperparam_optimized = get_results_hyperparam_optimized(results, hyperparameters)

###
duipi = results[results['method'] == 'DUIPI']
duipi_epsilon = duipi[duipi['epsilon_baseline'] == 0.5]
duipi_epsilon_traj = duipi_epsilon[duipi_epsilon['length_trajectory'] == 2000]
duipi_epsilon_traj['method_perf'].unique()
###
ramdp = results[results['method'] == 'RaMDP']
ramdp_epsilon_traj = ramdp[
    (ramdp['hyperparam'] == 0.001) & (ramdp['epsilon_baseline'] == 0.1) & (ramdp['length_trajectory'] == 2000)]

ramdp_epsilon_traj['seed'].iloc[0]

### Hyperparameter comparisons ###
row = 'epsilon_baseline'
col = 'length_trajectory'
y = 'method_perf'
results_hyperparam_optimized_list = []
# 1-Step-Exact-Soft-SPIBB-Hoeffding
method = '1-Step-Exact-Soft-SPIBB-Hoeffding'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col=col, row=row, hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, y)
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)

baselines = results[row].unique()
for i, ax_row in enumerate(g.axes):
    baseline = results['pi_b_perf'][results[row] == baselines[i]].iloc[0]
for ax in ax_row:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=-45,
                fontsize='small')
        # Remove the original text
        ax.texts[0].remove()
        #  ax.axvline(0, 0, 1, color='black')
    ax.axvline(baseline, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 0.2)])
# 1-Step-Approx-Soft-SPIBB-Hoeffding
method = '1-Step-Approx-Soft-SPIBB-Hoeffding'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=-45,
                fontsize='small')
        # Remove the original text
        ax.texts[0].remove()
    ax.axvline(0, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 0.5)])
# Exact-Soft-SPIBB-Hoeffding
method = 'Exact-Soft-SPIBB-Hoeffding'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 0.5)])
# Approx-Soft-SPIBB-Hoeffding
method = 'Approx-Soft-SPIBB-Hoeffding'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

del results_hyperparam_optimized_list[-1]
results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 1)])
# Approx-Soft-SPIBB-e_min
method = 'Approx-Soft-SPIBB-e_min'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 1)])
# Adv-Approx-Soft_SPIBB-Hoeffding
method = 'Adv-Approx-Soft_SPIBB-Hoeffding'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 1)])
# Adv-Approx-Soft_SPIBB-e_min
method = 'Adv-Approx-Soft_SPIBB-e_min'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='epsilon')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 1)])
# DUIPI
method = 'DUIPI'
g = sns.FacetGrid(data=results[results['method'] == method],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf')
g.add_legend(title='xi')
plt.subplots_adjust(top=0.95)
g.fig.suptitle(method)
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

results_hyperparam_optimized_list.append(results[(results['method'] == method) & (results['hyperparam'] == 0.1)])

### Intra batch_rl_algorithms comparison ###
results_hyperparam_optimized = pd.concat(results_hyperparam_optimized_list)
# results_hyperparam_optimized['method'] = results_hyperparam_optimized['method'].cat.remove_unused_categories()
# results_hyperparam_optimized['baseline_target_perf_ratio'] = results_hyperparam_optimized[
#    'baseline_target_perf_ratio'].round(decimals=1)
results_hyperparam_optimized_soft_spibb = results_hyperparam_optimized[
    results_hyperparam_optimized['method'] != 'DUIPI']
# results_hyperparam_optimized_soft_spibb[
#    'method'] = results_hyperparam_optimized_soft_spibb['method'].cat.remove_unused_categories()
# Soft-SPIBB comparison
g = sns.FacetGrid(data=results_hyperparam_optimized_soft_spibb,
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='method', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf', palette="tab10")
g.add_legend()
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Soft-SPIBB performance comparison')
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=-45,
                fontsize='small')
        # Remove the original text
        ax.texts[0].remove()
    ax.axvline(0, 0, 1, color='black')

row = 'epsilon_baseline'
col = 'length_trajectory'
g = sns.FacetGrid(data=results_hyperparam_optimized_soft_spibb,
                  row=row, col=col, hue='method', margin_titles=True)
g.map(sns.ecdfplot, 'method_perf', palette="tab10")
g.add_legend()
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Soft-SPIBB performance comparison')
baselines = results[row].unique()
for i, ax in enumerate(g.axes.flat):
    baseline = results_hyperparam_optimized_soft_spibb['pi_b_perf'][
        results_hyperparam_optimized_soft_spibb[row] == baselines[i]].iloc[0]
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=-45,
                fontsize='small')
        # Remove the original text
        ax.texts[0].remove()
    ax.axvline(baseline, 0, 1, color='black')

# Soft-SPIBB comparison, without Exact-Soft-SPIBB as its bad performance skrewed the plot
data = results_hyperparam_optimized_soft_spibb[
    (results_hyperparam_optimized_soft_spibb['method'] != '1-Step-Exact-Soft-SPIBB-Hoeffding') &
    (results_hyperparam_optimized_soft_spibb['method'] != 'Exact-Soft-SPIBB-Hoeffding')]
data['method'] = data['method'].cat.remove_unused_categories()
g = sns.FacetGrid(data=data,
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='method', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf', palette="tab10")
g.add_legend()
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Soft-SPIBB performance comparison, without Exact-Soft-SPIBB')
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

# Soft-SPIBB with basic rl
results_hyperparam_optimized_soft_spibb_with_basic_rl = pd.concat(
    [results_hyperparam_optimized_soft_spibb, results[results['method'] == 'Basic_rl']])
g = sns.FacetGrid(data=results_hyperparam_optimized_soft_spibb_with_basic_rl[
    ~ results_hyperparam_optimized_soft_spibb_with_basic_rl['method'].isin(
        ['Adv-Approx-Soft_SPIBB-e_min', 'Approx-Soft-SPIBB-e_min'])],
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='method', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf', palette="tab10")
g.add_legend()
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Soft-SPIBB performance comparison with basic rl, without e_min batch_rl_algorithms')
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

# All batch_rl_algorithms
results_hyperparam_optimized_all = pd.concat([results_hyperparam_optimized, results[
    results['method'].isin(['Basic_rl', 'R_min', 'MBIE', 'RaMDP', 'Pi_b_spibb', 'Pi_<b_SPIBB'])]])
g = sns.FacetGrid(data=results_hyperparam_optimized_all,
                  col='baseline_target_perf_ratio', row='nb_trajectories', hue='method', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf', palette="tab10")
g.add_legend()
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Performance comparison')
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

# Now just skip some
# results['method'].head(20).values to get the names of the methods
methods_to_skip = ['Adv-Approx-Soft_SPIBB-e_min', 'Approx-Soft-SPIBB-e_min', 'MBIE']
# methods_to_skip.append('Adv-Approx-Soft_SPIBB-e_min', 'Approx-Soft-SPIBB-e_min')  # Don't add any new information
data = results_hyperparam_optimized_all[~ results_hyperparam_optimized_all['method'].isin(methods_to_skip)]
g = sns.FacetGrid(
    data=data,
    col='baseline_target_perf_ratio', row='nb_trajectories', hue='method', margin_titles=True)
g.map(sns.ecdfplot, 'normalized_perf', palette="tab10")
g.add_legend()
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Performance comparison')
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    # Make title more human-readable and larger
    # if ax.get_title():
    #    ax.set_title(ax.get_title().split('=')[1],
    #                 fontsize='small')

    # Make right ylabel more human-readable and larger
    # Only the 2nd and 4th axes have something in ax.texts
    if
ax.texts:
# This contains the right ylabel text
txt = ax.texts[0]
ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
        txt.get_text(),
        transform=ax.transAxes,
        va='center',
        rotation=-45,
        fontsize='small')
# Remove the original text
ax.texts[0].remove()

ax.axvline(0, 0, 1, color='black')

# Lineplot of baseline_target_perf=0.9: nb_trajectories vs mean performance
g = sns.lineplot(
    data=results_hyperparam_optimized_all[~ results_hyperparam_optimized_all['method'].isin(methods_to_skip)][
        results_hyperparam_optimized_all['baseline_target_perf_ratio'] == 0.9], x='nb_trajectories',
    y='normalized_perf', hue='method', ci=0, markers=True, style='method', dashes=False, markersize=10)
g.set_title('Mean performance for baseline_target_perf_ratio 0.9')
# Lineplot of baseline_target_perf=0.9: nb_trajectories vs 1% CVaR performance
# If retyped, you run into some trouble here
int_cols = ['seed', 'iteration', 'nb_trajectories']
float_cols = ['softmax_target_perf_ratio', 'baseline_target_perf_ratio', 'baseline_perf', 'pi_rand_perf',
              'pi_star_perf', 'hyperparam', 'method_perf', 'time', 'normalized_perf']
category_cols = ['seed', 'nb_trajectories', 'softmax_target_perf_ratio', 'baseline_target_perf_ratio', 'method']
results_hyperparam_optimized_all[int_cols] = results_hyperparam_optimized_all[int_cols].astype('int64')
results_hyperparam_optimized_all[float_cols] = results_hyperparam_optimized_all[float_cols].astype('float64')
results_hyperparam_optimized_all[category_cols] = results_hyperparam_optimized_all[category_cols].astype('category')
results_hyperparam_optimized_all['method'] = results_hyperparam_optimized_all['method'].astype('object')
results_hyperparam_optimized_all_cvar = results_hyperparam_optimized_all.groupby(
    by=['baseline_target_perf_ratio', 'nb_trajectories', 'method'], as_index=False).quantile(0.01)
g = sns.lineplot(
    data=results_hyperparam_optimized_all_cvar[
        ~ results_hyperparam_optimized_all_cvar['method'].isin(methods_to_skip)][
        results_hyperparam_optimized_all_cvar['baseline_target_perf_ratio'] == 0.9], markersize=10,
    x='nb_trajectories', y='normalized_perf', hue='method', ci=0, markers=True, style='method', dashes=False)
g.set_title('1% CVaR performance for baseline_target_ratio 0.9')

### Heatmaps ###

# Heatmaps of all batch_rl_algorithms for all hyperparameters (mean)
path_heatmaps = os.path.join(path_results, 'Random_mdps_malicious', 'Figures', 'Intra batch_rl_algorithms comparisons',
                             'Heatmaps')
path_heatmaps_mean = os.path.join(path_heatmaps, 'mean')
methods = results['method'].unique()
for method in methods:
    cm_df_method = results[results['method'] == method]
hyperparams = cm_df_method['hyperparam'].unique()
if len(hyperparams) == 1:
    results_method_grouped_cm = cm_df_method.groupby(
        by=['baseline_target_perf_ratio', 'nb_trajectories'], as_index=False).mean()
results_method_grouped_cm_pivot = results_method_grouped_cm.pivot(
    index='baseline_target_perf_ratio', columns='nb_trajectories',
    values='normalized_perf')
plt.figure(figsize=(10, 10))
plt.title(f'Heatmap normalized mean performance for {method}.png')
heatmap = sns.heatmap(results_method_grouped_cm_pivot, vmin=-1, vmax=1, cmap="Spectral")
fig = heatmap.get_figure()
method = method.replace('<', '_smaller_')
filename = f'Heatmap {method} mean performance'
fig.savefig(os.path.join(path_heatmaps_mean, filename))
else:
for hyperparam in hyperparams:
    cm_df_hyperparam = cm_df_method[cm_df_method['hyperparam'] == hyperparam]
results_method_hyperparam_grouped_cvar = cm_df_hyperparam.groupby(
    by=['baseline_target_perf_ratio', 'nb_trajectories'], as_index=False).mean()
results_method_hyperparam_grouped_cm_pivot = results_method_hyperparam_grouped_cvar.pivot(
    index='baseline_target_perf_ratio', columns='nb_trajectories',
    values='normalized_perf')
plt.figure(figsize=(10, 10))
print(method)
plt.title(f'Heatmap normalized mean performance for {method} with hyperparam {hyperparam}')
heatmap = sns.heatmap(results_method_hyperparam_grouped_cm_pivot, vmin=-1, vmax=1, cmap="Spectral")
fig = heatmap.get_figure()
method = method.replace('<', '_smaller_')
filename = f'Heatmap {method} hyperparam {hyperparam} mean performance.png'
fig.savefig(os.path.join(path_heatmaps_mean, filename))

# Heatmaps of all batch_rl_algorithms for all hyperparameters (Cvar)
path_heatmaps = os.path.join(path_results, 'Random_mdps_malicious', 'Figures', 'Intra batch_rl_algorithms comparisons',
                             'Heatmaps')
path_heatmaps_cm = os.path.join(path_heatmaps, 'Cvar')
methods = results['method'].unique()
for method in methods:
    cm_df_method = results[results['method'] == method]
hyperparams = cm_df_method['hyperparam'].unique()
if len(hyperparams) == 1:
    results_method_grouped_cm = cm_df_method[
        ['baseline_target_perf_ratio', 'nb_trajectories', 'normalized_perf']].groupby(
        by=['baseline_target_perf_ratio', 'nb_trajectories'], as_index=False).quantile(0.01)
results_method_grouped_cm_pivot = results_method_grouped_cm.pivot(
    index='baseline_target_perf_ratio', columns='nb_trajectories',
    values='normalized_perf')
plt.figure(figsize=(10, 10))
plt.title(f'Heatmap normalized 1% Cvar performance for {method}.png')
heatmap = sns.heatmap(results_method_grouped_cm_pivot, vmin=-1, vmax=1, cmap="Spectral")
fig = heatmap.get_figure()
method = method.replace('<', '_smaller_')
filename = f'Heatmap {method} 1% Cvar performance'
fig.savefig(os.path.join(path_heatmaps_cm, filename))
else:
for hyperparam in hyperparams:
    cm_df_hyperparam = cm_df_method[cm_df_method['hyperparam'] == hyperparam]
results_method_hyperparam_grouped_cvar = cm_df_hyperparam[
    ['baseline_target_perf_ratio', 'nb_trajectories', 'normalized_perf']].groupby(
    by=['baseline_target_perf_ratio', 'nb_trajectories'], as_index=False).quantile(0.01)
results_method_hyperparam_grouped_cm_pivot = results_method_hyperparam_grouped_cvar.pivot(
    index='baseline_target_perf_ratio', columns='nb_trajectories',
    values='normalized_perf')
plt.figure(figsize=(10, 10))
print(method)
plt.title(f'Heatmap normalized 1% Cvar performance for {method} with hyperparam {hyperparam}')
heatmap = sns.heatmap(results_method_hyperparam_grouped_cm_pivot, vmin=-1, vmax=1, cmap="Spectral")
fig = heatmap.get_figure()
method = method.replace('<', '_smaller_')
filename = f'Heatmap {method} hyperparam {hyperparam} 1% Cvar performance.png'
fig.savefig(os.path.join(path_heatmaps_cm, filename))

### Critical mass analysis ###
methods = results['method'].unique()
hyperparams = results['hyperparam'].unique()
ratios = results['baseline_target_perf_ratio'].unique()
nb_trajectories_list = results['nb_trajectories'].unique()
cm_list = []
warning = ''
for ratio in ratios:
    for
nb_trajectories in nb_trajectories_list:
for method in methods:
    results_specific = results[
        (results['method'] == method) & (results['nb_trajectories'] == nb_trajectories)
        & (results['baseline_target_perf_ratio'] == ratio)]
hyperparams = results_specific['hyperparam'].unique()
for hyperparam in hyperparams:
    print(f'Ratio: {ratio}; nb_trajectories: {nb_trajectories}; Method: {method}; Hyperparam: {hyperparam}')
# cm = critical_mass(results_specific[results_specific['hyperparam'] == hyperparam])
if method == 'Basic_rl':
    df = results_specific
else:
    df = results_specific[results_specific['hyperparam'] == hyperparam]
cm = sum(df['normalized_perf'].clip(upper=0)) / df.shape[0]
cm_list.append([ratio, nb_trajectories, method, hyperparam, cm])

if df.shape[0] != 9107:
    warning += f'Ratio: {ratio}; nb_trajectories: {nb_trajectories}; Method: {method};' \
               f' Hyperparam: {hyperparam}; Length: {df.shape[0]} \n'

cm_df = pd.DataFrame(cm_list,
                     columns=['baseline_target_perf_ratio', 'nb_trajectories', 'method', 'hyperparam', 'critical_mass'])
cm_df.to_csv(os.path.join(path_results, 'Random_mdps_easter', 'Experiments', 'critical_mass.csv'))

# Heatmaps of all batch_rl_algorithms for all hyperparameters (critical mass)
path_heatmaps = os.path.join(path_results, 'Random_mdps_easter', 'Figures', 'Intra batch_rl_algorithms comparisons',
                             'Heatmaps')
path_heatmaps_cm = os.path.join(path_heatmaps, 'critical_mass')
methods = cm_df['method'].unique()
for method in methods:
    cm_df_method = cm_df[cm_df['method'] == method]
hyperparams = cm_df_method['hyperparam'].unique()
if len(hyperparams) == 1:
    results_method_grouped_cm_pivot = cm_df_method.pivot(
        index='baseline_target_perf_ratio', columns='nb_trajectories',
        values='critical_mass')
plt.figure(figsize=(10, 10))
plt.title(f'Heatmap critical mass for {method}.png')
heatmap = sns.heatmap(results_method_grouped_cm_pivot, vmin=-0.05, vmax=0, cmap="Spectral", annot=True)
fig = heatmap.get_figure()
method = method.replace('<', '_smaller_')
filename = f'Heatmap {method} critical mass'
fig.savefig(os.path.join(path_heatmaps_cm, filename))
else:
for hyperparam in hyperparams:
    cm_df_hyperparam = cm_df_method[cm_df_method['hyperparam'] == hyperparam]
results_method_hyperparam_grouped_cm_pivot = cm_df_hyperparam.pivot(
    index='baseline_target_perf_ratio', columns='nb_trajectories',
    values='critical_mass')
plt.figure(figsize=(10, 10))
print(method)
plt.title(f'Heatmap critical mass for {method} with hyperparam {hyperparam}')
heatmap = sns.heatmap(results_method_hyperparam_grouped_cm_pivot, vmin=-0.05, vmax=0, cmap="Spectral",
                      annot=True)
fig = heatmap.get_figure()
method = method.replace('<', '_smaller_')
filename = f'Heatmap {method} hyperparam {hyperparam} critical mass.png'
fig.savefig(os.path.join(path_heatmaps_cm, filename))

########################
path_results = r'C:\Users\phili\PycharmProjects\safe_rl\results\exp_perf'

results = pd.read_excel(os.path.join(path_results, 'test_adv_090920', 'results_1234.xlsx'), index_col=0)

results.loc[:, ['perfrl', 'perf_soft_SPIBB_sort_Q', 'perf_adv_soft_SPIBB_sort_Q']].plot()
plt.title('Mean Performance for ratio=0.9, epsilon=2')
plt.xlabel('Number experiment')
plt.ylabel('Performance')

results_method_hyperparam_grouped_cvar = results[
    (results['epsilon'] == 2.0) & (results['baseline_target_perf_ratio'] == 0.9)].groupby(
    by=['nb_trajectories']).mean()

results_method_hyperparam_grouped_cvar.loc[:,
['perfrl', 'perf_Pi_b_SPIBB', 'perf_Pi_leq_b_SPIBB', 'perf_soft_SPIBB_simplex', 'perf_soft_SPIBB_sort_Q',
 'perf_adv_soft_SPIBB_sort_Q', 'perf_soft_SPIBB_simplex_1step', 'perf_soft_SPIBB_sort_Q_1step']].plot()

results_method_hyperparam_grouped_cvar.mean()

results_grouped_min = results[(results['epsilon'] == 2.0) & (results['baseline_target_perf_ratio'] == 0.9)].groupby(
    by=['nb_trajectories']).min()

results_grouped_min.loc[:,
['perfrl', 'perf_Pi_b_SPIBB', 'perf_Pi_leq_b_SPIBB', 'perf_soft_SPIBB_simplex', 'perf_soft_SPIBB_sort_Q',
 'perf_adv_soft_SPIBB_sort_Q', 'perf_soft_SPIBB_simplex_1step', 'perf_soft_SPIBB_sort_Q_1step']].plot()

sum(results['perf_soft_SPIBB_sort_Q_1step'] > results['perf_adv_soft_SPIBB_sort_Q'])

# Only when the changes should be really saved
# results.to_excel(os.path.join(path_own_experiments, 'test_adv_090920', 'results_1234.xlsx'))

### Helicopter ###

path = r'C:\Users\phili\PycharmProjects\SPIBB-DQN\baseline\helicopter_env\dataset\1000\123\1_0'

# See SPIBB-DQN\experiment.py line 245
column_names = ['epoch', 'eval_scores/eval_episodes', 'eval_steps/eval_episodes', 'eval_episodes']
list_of_experiments = []
for i in range(15):
    file_name = f'spibb_{i}_10.0.csv'
df = pd.read_csv(os.path.join(path, file_name), header=None, names=column_names)
df['Experiment number'] = [i] * df.shape[0]
list_of_experiments.append(df)

experiments = pd.concat(list_of_experiments)

print(np.mean(experiments['eval_scores/eval_episodes']))

#####

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

import exploration_utils
from exploration_utils import hyperparams_dict

config = configparser.ConfigParser()
config.read(os.path.join(r'C:\Users\phili\PycharmProjects\safe_rl', 'config.ini'))
path_results = config['PATHS']['results_path']

random_mdp = False
if random_mdp:
    directory = os.path.join(path_results, 'Random_mdps', 'Random_mdps_easter')
else:
    directory = os.path.join(path_results, 'wet_chicken', 'heuristic')
exp_path = os.path.join(directory, 'Experiments')
fig_path = os.path.join(directory, 'Figures')

if random_mdp:
    col = 'baseline_target_perf_ratio'
    row = 'nb_trajectories'
    y = 'normalized_perf'
else:
    row = 'epsilon_baseline'
    col = 'length_trajectory'
    y = 'method_perf'

results = pd.read_csv(os.path.join(exp_path, 'results_hyperparam_optimized.csv'))

results_grouped = results.groupby(by=[row, col, 'method'], as_index=False).var()

