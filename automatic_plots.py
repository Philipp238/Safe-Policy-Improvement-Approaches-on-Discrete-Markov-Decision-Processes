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

if __name__ == '__main__':
    random_mdps = True
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

    # Hyperparameter ECDF plot
    results = pd.read_csv(os.path.join(exp_path, 'results_full_normalized.csv'))
    results['normalized_perf'] = (results['method_perf'] - results['baseline_perf']) / (
            results['pi_star_perf'] - results['baseline_perf'])
    results.to_csv(os.path.join(exp_path, 'results_full_normalized.csv'), index=False)

    if random_mdps:
        right = 0.82
    else:
        right = 0.7
    reduced = True
    methods = ['RaMDP']
    exploration_utils.hyperparameter_ecdf(results, fig_path, col=row, row=col, y=y, title_size=20, text_size=16,
                                          height=2, aspect=2.1, columns_row_titles_short=False, right=right,
                                          methods=methods, reduced=reduced, random_mdps=random_mdps)

    # # Hyperparameter optimized ECDF plot SPIBB
    # results = pd.read_csv(os.path.join(exp_path, 'results_soft_spibb_hoeffding_mpeb_full_normalized.csv'))
    # exploration_utils.hyperparameter_optimized_ecdf(results, fig_path, title='Soft SPIBB optimized comparisons',
    #                                                 row=row, col=col, y=y)

    # # Hyperparameter optimized line plots all algorithms
    # results = pd.read_csv(os.path.join(exp_path, 'soft_spibb_mpeb_hoeffding_hyperparam_optimized.csv'))
    # x = col
    # line_plot_path = os.path.join(fig_path, 'Intra algorithms comparisons', 'Lineplot')
    # file_appendix = ''
    # # methods_to_skip = ['Basic_rl', 'DUIPI_bayesian', 'DUIPI_frequentist', 'Lower-SPIBB', 'R_min', 'RaMDP', 'SPIBB']
    # # methods_to_skip = ['1-Step-Approx-Soft-SPIBB_mpeb', 'Adv-Approx-Soft-SPIBB_mpeb', 'Approx-Soft-SPIBB_mpeb',
    # #                    'Lower-Approx-Soft-SPIBB_mpeb', 'DUIPI_frequentist']
    # # methods_to_skip = ['Basic_rl']
    # methods_to_skip = []
    # # y_range = {'mean': -0.1, 'cvar': -0.5}
    # y_range = None
    # # palette = None
    # palette = {'1-Step-Approx-Soft-SPIBB_mpeb': 'brown', '1-Step-Approx-Soft-SPIBB_hoeffding': 'darkblue',
    #            'Adv-Approx-Soft-SPIBB_mpeb': 'brown', 'Adv-Approx-Soft-SPIBB_hoeffding': 'darkblue',
    #            'Approx-Soft-SPIBB_mpeb': 'brown', 'Approx-Soft-SPIBB_hoeffding': 'darkblue',
    #            'Lower-Approx-Soft-SPIBB_mpeb': 'brown', 'Lower-Approx-Soft-SPIBB_hoeffding': 'darkblue'}
    # # markers = True
    # markers = {'1-Step-Approx-Soft-SPIBB_mpeb': '*', '1-Step-Approx-Soft-SPIBB_hoeffding': '*',
    #           'Adv-Approx-Soft-SPIBB_mpeb': '^', 'Adv-Approx-Soft-SPIBB_hoeffding': '^',
    #           'Approx-Soft-SPIBB_mpeb': 'P', 'Approx-Soft-SPIBB_hoeffding': 'P',
    #           'Lower-Approx-Soft-SPIBB_mpeb': '8', 'Lower-Approx-Soft-SPIBB_hoeffding': '8'}
    # exploration_utils.automated_hyperparameter_optimized_line_plot(results, col, row, x, y, palette=palette,
    #                                                                markers=markers,
    #                                                                random_mdps=random_mdps, y_range=y_range,
    #                                                                fig_path=fig_path, file_appendix=file_appendix,
    #                                                                methods_to_skip=methods_to_skip, cvar=True)

    # methods_of_interest = ['1-Step-Approx-Soft-SPIBB-Hoeffding',
    #                        'Approx-Soft-SPIBB-Hoeffding',
    #                        'Adv-Approx-Soft_SPIBB-Hoeffding', 'Basic_rl', 'DUIPI']
    # results = results[results['method'].isin(methods_of_interest)]
    # hyperparameter_optimized_ecdf(results, fig_path, title='Soft SPIBB & DUIPI optimized batch_rl_algorithms comparisons', row=row,
    #                               col=col, y=y)
    # x = col
    # epsilons_baseline = results['epsilon_baseline'].unique()
    # exploration_utils.automated_hyperparameter_optimized_line_plot(results, col, row, x, y, epsilons_baseline,
    #                                                                methods_to_skip=[], fig_path=fig_path)
