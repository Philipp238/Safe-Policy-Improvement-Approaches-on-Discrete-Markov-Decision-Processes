import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

from exploration_utils import critical_mass
from exploration_utils import get_results_hyperparam_optimized
from exploration_utils import runtime_ecdf
from exploration_utils import load_data_first_time

plt.ioff()


config = configparser.ConfigParser()
config.read(os.path.join(r'C:\Users\phili\PycharmProjects\safe_rl', 'config.ini'))
path_results = config['PATHS']['results_path']

safety_path = os.path.join(path_results, 'wet_chicken', 'safety')
exp_path = os.path.join(safety_path, 'Experiments')
first_time = False
skip_small_data = True
if ~ first_time:
    results = pd.read_csv(os.path.join(exp_path, 'results.csv'))
    methods = results['method'].unique()
    deltas = results['delta'].unique()
else:
    results = load_data_first_time(exp_path=exp_path)
    methods = results['method'].unique()
    deltas = results['delta'].unique()

    results['advantage'] = None
    soft_spibb_mask = results['method'].isin(methods[:4])
    gamma = 0.95
    results.loc[soft_spibb_mask, 'advantage'] = results[soft_spibb_mask]['hyperparam'] / (1 - gamma) + \
                                                results[soft_spibb_mask][
                                                    'bound']

    results.loc[soft_spibb_mask, 'bound'] = - results[soft_spibb_mask]['hyperparam'] / (1 - gamma)

    results.to_csv(os.path.join(exp_path, 'results.csv'))

if skip_small_data:
    results = results[results['length_trajectory'] >= 50000]

# # Now find out which ones are actually safe, so we do ECDFs for the bounds and performance at the same time
# row = 'length_trajectory'
# col = 'epsilon_baseline'
# y = 'method_perf'
#
# for method in methods[2:]:
#     for delta in deltas:
#         results_method_delta = results[(results['method'] == method) & (results['delta'] == delta)]
#
#         fig, ax = plt.subplots()
#         fig.set_size_inches(20, 15)
#
#         g = sns.FacetGrid(data=results_method_delta, col=col, row=row, margin_titles=True)
#         g.map(sns.ecdfplot, 'bound', color='red')
#         g.map(sns.ecdfplot, y)
#
#         g.fig.suptitle(f'Actual performance and the bound on performance for delta {delta} for {method}')
#         plt.subplots_adjust(top=0.9, right=0.85)
#         plt.legend(title='legend', bbox_to_anchor=(1.05, 1), labels=['bound on performance', 'actual performance'])
#         baselines = results[row].unique()
#         for i, ax_row in enumerate(g.axes):
#             if y != 'normalized_perf':
#                 baseline = results['pi_b_perf'][results[row] == baselines[i]].iloc[0]
#             for ax in ax_row:
#                 # Make x and y-axis labels slightly larger
#                 # ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
#                 # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')
#
#                 # Make title more human-readable and larger
#                 # if ax.get_title():
#                 #    ax.set_title(ax.get_title().split('=')[1],
#                 #                 fontsize='small')
#
#                 # Make right ylabel more human-readable and larger
#                 # Only the 2nd and 4th axes have something in ax.texts
#                 if ax.texts:
#                     # This contains the right ylabel text
#                     txt = ax.texts[0]
#                     ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
#                             txt.get_text(),
#                             transform=ax.transAxes,
#                             va='center',
#                             rotation=-45,
#                             fontsize='small')
#                     # Remove the original text
#                     ax.texts[0].remove()
#
#                 if y == 'normalized_perf':
#                     ax.axvline(0, 0, 1, color='black')
#                 else:
#                     ax.axvline(baseline, 0, 1, color='black')
#         if skip_small_data:
#             appendix = '_reduced'
#         else:
#             appendix = ''
#
#         g.savefig(os.path.join(safety_path, 'Figures',
#                                'Hyperparameter comparisons', 'ECDF', f'{method}_delta_{delta}{appendix}.png'))

G_max = 40  # need G_max instead of V_max

# vertical
row = 'length_trajectory'
col = 'epsilon_baseline'
y = 'method_perf'
for method in methods[4:]:
    print(f'Starting with {method}.')
    for delta in deltas:
        print(f'Starting with {delta} in {deltas}.')
        results_method_delta = results[(results['method'] == method) & (results['delta'] == delta)]

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 15)

        g = sns.FacetGrid(data=results_method_delta, col=col, row=row, margin_titles=True)
        g.map(sns.ecdfplot, 'bound', color='red')
        g.map(sns.ecdfplot, y)

        g.fig.suptitle(f'Actual performance and the bound on performance for delta {delta} for {method}')
        plt.subplots_adjust(top=0.9, right=0.85)
        plt.legend(title='legend', bbox_to_anchor=(1.05, 1), labels=['bound on performance', 'actual performance'])
        baseline_name = 'epsilon_baseline'
        baselines = results[baseline_name].unique()
        for ax_row in g.axes:
            for i, ax in enumerate(ax_row):
                if y != 'normalized_perf':
                    baseline = results['pi_b_perf'][results[baseline_name] == baselines[i]].iloc[0]
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

                if y == 'normalized_perf':
                    ax.axvline(0, 0, 1, color='black')
                else:
                    ax.axvline(baseline, 0, 1, color='black')
        if skip_small_data:
            appendix = '_reduced'
        else:
            appendix = ''

        g.savefig(os.path.join(safety_path, 'Figures',
                               'Hyperparameter comparisons', 'ECDF', f'{method}_delta_{delta}{appendix}.png'))

for method in methods[:4]:
    print(f'Starting with {method}.')
    for delta in deltas:
        print(f'Starting with {delta} in {deltas}.')
        results_method_delta = results[(results['method'] == method) & (results['delta'] == delta)]
        epsilons = results_method_delta['hyperparam'].unique()
        for epsilon in epsilons:
            print(f'Starting with {epsilon} in {epsilons}.')
            results_method_delta_epsilon = results_method_delta[results_method_delta['hyperparam'] == epsilon]
            results_method_delta_epsilon['bound_by_epsilon'] = results_method_delta_epsilon['pi_b_perf'] - epsilon \
                                                               / (1 - results_method_delta_epsilon['gamma']) * G_max
            fig, ax = plt.subplots()
            fig.set_size_inches(20, 15)

            g = sns.FacetGrid(data=results_method_delta_epsilon, col=col, row=row, margin_titles=True)
            g.map(sns.ecdfplot, 'bound_by_epsilon', color='red')
            g.map(sns.ecdfplot, y)

            g.fig.suptitle(
                f'Actual performance and the bound on performance for delta {delta} for {method} with epsilon {epsilon}')
            plt.subplots_adjust(top=0.9, right=0.85)
            plt.legend(title='legend', bbox_to_anchor=(1.05, 1), labels=['bound on performance', 'actual performance'])
            baseline_name = 'epsilon_baseline'
            baselines = results[baseline_name].unique()
            for ax_row in g.axes:
                for i, ax in enumerate(ax_row):
                    if y != 'normalized_perf':
                        baseline = results['pi_b_perf'][results[baseline_name] == baselines[i]].iloc[0]
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

                    if y == 'normalized_perf':
                        ax.axvline(0, 0, 1, color='black')
                    else:
                        ax.axvline(baseline, 0, 1, color='black')

            if skip_small_data:
                appendix = '_reduced'
            else:
                appendix = ''

            g.savefig(os.path.join(safety_path, 'Figures', 'Hyperparameter comparisons',
                                   'ECDF', f'{method}_epsilon_{epsilon}_delta_{delta}{appendix}.png'))

# horizontal
row = 'epsilon_baseline'
col = 'length_trajectory'
y = 'method_perf'
for method in methods[4:]:
    for delta in deltas:
        results_method_delta = results[(results['method'] == method) & (results['delta'] == delta)]

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 15)

        g = sns.FacetGrid(data=results_method_delta, col=col, row=row, margin_titles=True)
        g.map(sns.ecdfplot, 'bound', color='red')
        g.map(sns.ecdfplot, y)

        g.fig.suptitle(f'Actual performance and the bound on performance for delta {delta} for {method}')
        plt.subplots_adjust(top=0.9, right=0.85)
        plt.legend(title='legend', bbox_to_anchor=(1.05, 1), labels=['bound on performance', 'actual performance'])
        baseline_name = 'epsilon_baseline'
        baselines = results[baseline_name].unique()
        for i, ax_row in enumerate(g.axes):
            if y != 'normalized_perf':
                baseline = results['pi_b_perf'][results[baseline_name] == baselines[i]].iloc[0]
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

                if y == 'normalized_perf':
                    ax.axvline(0, 0, 1, color='black')
                else:
                    ax.axvline(baseline, 0, 1, color='black')

        g.savefig(
            os.path.join(safety_path, 'Figures', 'Hyperparameter comparisons', 'ECDF', f'{method}_delta_{delta}.png'))
        plt.close()

for method in methods[:4]:
    for delta in deltas:
        results_method_delta = results[(results['method'] == method) & (results['delta'] == delta)]
        epsilons = results_method_delta['hyperparam'].unique()
        for epsilon in epsilons:
            results_method_delta_epsilon = results_method_delta[results_method_delta['hyperparam'] == epsilon]
            results_method_delta_epsilon['bound_by_epsilon'] = results_method_delta_epsilon['pi_b_perf'] - epsilon \
                                                               / (1 - results_method_delta_epsilon['gamma']) * G_max
            fig, ax = plt.subplots()
            fig.set_size_inches(20, 15)

            g = sns.FacetGrid(data=results_method_delta_epsilon, col=col, row=row, margin_titles=True)
            g.map(sns.ecdfplot, 'bound_by_epsilon', color='red')
            g.map(sns.ecdfplot, y)

            g.fig.suptitle(
                f'Actual performance and the bound on performance for delta {delta} for {method} with epsilon {epsilon}')
            plt.subplots_adjust(top=0.9, right=0.85)
            plt.legend(title='legend', bbox_to_anchor=(1.05, 1), labels=['bound on performance', 'actual performance'])
            baseline_name = 'epsilon_baseline'
            baselines = results[baseline_name].unique()
            for i, ax_row in enumerate(g.axes):
                if y != 'normalized_perf':
                    baseline = results['pi_b_perf'][results[baseline_name] == baselines[i]].iloc[0]
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

                    if y == 'normalized_perf':
                        ax.axvline(0, 0, 1, color='black')
                    else:
                        ax.axvline(baseline, 0, 1, color='black')

            g.savefig(os.path.join(safety_path, 'Figures', 'Hyperparameter comparisons', 'ECDF',
                                   f'{method}_epsilon_{epsilon}_delta_{delta}.png'))
            plt.close()

results.to_csv(os.path.join(exp_path, 'results.csv'))
