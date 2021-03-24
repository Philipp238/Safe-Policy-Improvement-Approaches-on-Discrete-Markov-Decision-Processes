import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

hyperparams_dict = {
    'RaMDP': 'kappa',
    'R_min': 'N_wedge',
    'MBIE': 'delta',
    'Pi_b_spibb': 'N_wedge',
    'SPIBB': 'N_wedge',
    'Lower-SPIBB': 'N_wedge',
    'Pi_<b_SPIBB': 'N_wedge',
    '1-Step-Exact-Soft-SPIBB-Hoeffding': 'epsilon',
    '1-Step-Approx-Soft-SPIBB-Hoeffding': 'epsilon',
    '1-Step-Approx-Soft-SPIBB_hoeffding': 'epsilon',
    '1-Step-Approx-Soft-SPIBB_mpeb': 'epsilon',
    'Exact-Soft-SPIBB-Hoeffding': 'epsilon',
    'Approx-Soft-SPIBB-Hoeffding': 'epsilon',
    'Approx-Soft-SPIBB_hoeffding': 'epsilon',
    'Approx-Soft-SPIBB_mpeb': 'epsilon',
    'Approx-Soft-SPIBB-e_min': 'epsilon',
    'Adv-Approx-Soft_SPIBB-Hoeffding': 'epsilon',
    'Adv-Approx-Soft-SPIBB_hoeffding': 'epsilon',
    'Adv-Approx-Soft-SPIBB_mpeb': 'epsilon',
    'Adv-Approx-Soft_SPIBB-e_min': 'epsilon',
    'Lower-Approx-Soft-SPIBB-e_min': 'epsilon',
    'Lower-Approx-Soft-SPIBB_hoeffding': 'epsilon',
    'Lower-Approx-Soft-SPIBB_mpeb': 'epsilon',
    'DUIPI': 'xi',
    'DUIPI_bayesian': 'xi',
    'DUIPI_frequentist': 'xi'
}


def critical_mass(df):
    return sum(df['normalized_perf'].clip(upper=0)) / df.shape[1]


def get_results_hyperparam_optimized(results, dict):
    results_hyperparam_optimized_list = [results[results['method'] == 'Basic_rl']]
    for method in dict.keys():
        results_hyperparam_optimized_list.append(
            results[(results['method'] == method) & (results['hyperparam'] == dict[method])])
    results_hyperparam_optimized = pd.concat(results_hyperparam_optimized_list)
    return results_hyperparam_optimized


def hyperparameter_ecdf(results, fig_path, col='baseline_target_perf_ratio', row='nb_trajectories',
                        y='normalized_perf', title_size=20, text_size=16, height=2, aspect=2.1, methods=None,
                        columns_row_titles_short=False, right=0.82, reduced=False, random_mdps=True):
    if reduced:
        columns_row_titles_short = True
        rotation = -45
        right = 0.6
        if random_mdps:
            results = results[results['baseline_target_perf_ratio'] == 0.9]
        else:
            results = results[results['epsilon_baseline'] == 0.1]
    else:
        rotation = -45
    if not methods:
        methods = results['method'].unique()
    for method in methods:
        print(f'ECDF of {method}')
        if len(results[results['method'] == method]['hyperparam'].unique()) > 1:
            g = sns.FacetGrid(data=results[results['method'] == method], height=height, aspect=aspect,
                              col=col, row=row, hue='hyperparam', margin_titles=True, legend_out=True)
            g.map(sns.ecdfplot, y)
            g.add_legend(title=hyperparams_dict[method], title_fontsize=text_size, fontsize=text_size)
            g.map(sns.ecdfplot, y)
            plt.setp(g._legend.get_title(), fontsize=text_size)
        else:
            g = sns.FacetGrid(data=results[results['method'] == method],
                              col=col, row=row, margin_titles=True)

        plt.subplots_adjust(top=0.9, right=right)
        g.fig.suptitle(method, size=title_size)
        if random_mdps:
            g.set_xticklabels([-3, -2, -1, 0, 1], size=14)
        else:
            g.set_xticklabels([10, 20, 30, 40], size=14)
        g.set_yticklabels(np.arange(6) / 5, size=14)

        baselines = results[row].unique()
        first = True
        for i, ax_row in enumerate(g.axes):
            if y != 'normalized_perf':
                baseline = results['pi_b_perf'][results[row] == baselines[i]].iloc[0]
            for ax in ax_row:
                # Make x and y-axis labels slightly larger
                ax.set_xlabel(ax.get_xlabel(), fontsize=text_size)
                ax.set_ylabel(ax.get_ylabel(), fontsize=text_size)

                if random_mdps:
                    ax.set_xlim(-3, 1)
                    ax.set_xticks([-3, -2, -1, 0, 1])
                else:
                    ax.set_xlim(10, 45)
                    ax.set_xticks([10, 20, 30, 40])

                # Make title more human-readable and larger
                # if ax.get_title():
                #    ax.set_title(ax.get_title().split('=')[1],
                #                 fontsize='small')

                if ax.get_title():
                    text_string = ax.get_title()
                    if text_string.startswith('baseline_target_perf_ratio'):
                        parts = text_string.split(' ')
                        parts[0] = 'behavior_perf_ratio'
                        text_string = ' '.join(parts)
                    if text_string.startswith('epsilon'):
                        parts = text_string.split(' ')
                        parts[0] = 'epsilon_behavior'
                        text_string = ' '.join(parts)
                    if columns_row_titles_short:
                        if first:
                            first = False
                        else:
                            text_string = text_string.split(' ')[2]
                    ax.set_title(text_string, fontsize=text_size)

                # Make right ylabel more human-readable and larger
                # Only the 2nd and 4th axes have something in ax.texts
                if ax.texts:
                    # This contains the right ylabel text
                    txt = ax.texts[0]
                    text_string = txt.get_text()
                    if columns_row_titles_short:
                        if i > 0:
                            text_string = text_string.split(' ')[2]
                    ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                            text_string,
                            transform=ax.transAxes,
                            va='center',
                            rotation=rotation,
                            fontsize=text_size)
                    # Remove the original text
                    ax.texts[0].remove()

                if y == 'normalized_perf':
                    ax.axvline(0, 0, 1, color='black')
                else:
                    ax.axvline(baseline, 0, 1, color='black')

        file_name = method.replace('<', '_smaller_')
        if reduced:
            file_name = file_name + '_reduced'
        g.savefig(os.path.join(fig_path, 'Hyperparameter comparisons', 'ECDF', file_name))


def hyperparameter_optimized_ecdf(results, fig_path, title, col='baseline_target_perf_ratio', row='nb_trajectories',
                                  y='normalized_perf'):
    g = sns.FacetGrid(data=results, row=row, col=col, hue='method', margin_titles=True)
    g.map(sns.ecdfplot, 'method_perf', palette="tab10")
    g.add_legend()
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle(title)
    baselines = results[row].unique()
    for i, ax_row in enumerate(g.axes):
        if y != 'normalized_perf':
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
            if y == 'normalized_perf':
                ax.axvline(0, 0, 1, color='black')
            else:
                ax.axvline(baseline, 0, 1, color='black')
    g.savefig(os.path.join(fig_path, 'Intra algorithms comparisons', 'ECDF', f'{title}.png'))


def automated_hyperparameter_optimized_line_plot(results, col, row, x, y, fig_path, random_mdps, palette=None,
                                                 markers=True, methods_to_skip=[], file_appendix='', cvar=True,
                                                 y_range=None):
    line_plot_path = os.path.join(fig_path, 'Intra algorithms comparisons', 'Lineplot')
    sns.set(font_scale=2)
    if random_mdps:
        baseline_parameter_name = 'baseline_target_perf_ratio'
        baseline_parameter_name_displayed = 'baseline_perf_ratio'
    else:
        baseline_parameter_name = 'epsilon_baseline'
        baseline_parameter_name_displayed = 'epsilon_baseline'
        results.drop(columns='baseline_method', inplace=True)
    baseline_parameters = results[baseline_parameter_name].unique()
    for baseline_parameter in baseline_parameters:
        print(f'Starting {baseline_parameter_name} {baseline_parameter} out of {baseline_parameters}.')
        title_cvar = f"1% CVaR performance for {baseline_parameter_name_displayed} {baseline_parameter}"
        path_cvar = os.path.join(line_plot_path, title_cvar + file_appendix + '.png')
        results_of_interest = results[results[baseline_parameter_name] == baseline_parameter]
        if random_mdps:
            altitude_horizontal_line = 0
        else:
            altitude_horizontal_line = results_of_interest['pi_b_perf'].iloc[0]
        if cvar:
            results_grouped = get_cvar(results_of_interest=results_of_interest, quantile=0.01, row=row, col=col,
                                       random_mdp=random_mdps)
        else:
            results_grouped = results_of_interest.groupby(by=[row, col, 'method'], as_index=False).quantile(0.01)
        if y_range:
            lower_limit_cvar = y_range['cvar']
            lower_limit_mean = y_range['mean']
        else:
            lower_limit_cvar = None
            lower_limit_mean = None
        hyperparameter_optimized_line_plot(results_grouped, x=x, y=y, lower_limit=lower_limit_cvar, markers=markers,
                                           title=title_cvar, path=path_cvar, methods_to_skip=methods_to_skip,
                                           altitude_horizontal_line=altitude_horizontal_line, palette=palette)

        title_mean = f"Mean performance for {baseline_parameter_name_displayed} {baseline_parameter}"
        path_mean = os.path.join(line_plot_path, title_mean + file_appendix + '.png')
        results_grouped = results_of_interest.groupby(by=[row, col, 'method'], as_index=False).mean()
        hyperparameter_optimized_line_plot(results_grouped, x=x, y=y, lower_limit=lower_limit_mean, markers=markers,
                                           title=title_mean, path=path_mean, methods_to_skip=methods_to_skip,
                                           altitude_horizontal_line=altitude_horizontal_line, palette=palette)


def get_cvar(results_of_interest, quantile, row, col, random_mdp):
    results_grouped_quantile = results_of_interest.groupby(by=[row, col, 'method'], as_index=False).quantile(quantile)
    methods = results_grouped_quantile['method'].unique()
    if random_mdp:
        trajectory_name = 'nb_trajectories'
    else:
        trajectory_name = 'length_trajectory'
    lengths_trajectories = results_grouped_quantile[trajectory_name].unique()
    results_list = []
    for method in methods:
        for length_trajectory in lengths_trajectories:
            upper_bound = results_grouped_quantile[
                (results_grouped_quantile['method'] == method) & (
                        results_grouped_quantile[trajectory_name] == length_trajectory)][
                'method_perf'].values[0]
            results_reduced_method_length_trajectory = results_of_interest[(results_of_interest['method'] == method) & (
                    results_of_interest[trajectory_name] == length_trajectory) & (
                                                                results_of_interest['method_perf'] <= upper_bound)]
            results_list.append(results_reduced_method_length_trajectory)
    results_reduced = pd.concat(results_list)
    results_grouped_cvar = results_reduced.groupby(by=[row, col, 'method'], as_index=False).mean()
    return results_grouped_cvar


def hyperparameter_optimized_line_plot(results, x, y, title, altitude_horizontal_line=None, methods_to_skip=[],
                                       path=None, lower_limit=None, palette=None, markers=True):
    fig, ax = plt.subplots()
    # the size of A4 paper
    fig.set_size_inches(20, 15)
    g = sns.lineplot(
        data=results[~ results['method'].isin(methods_to_skip)], markersize=17, palette=palette,
        x=x, y=y, hue='method', ci=0, style='method', dashes=False, ax=ax, markers=markers)
    g.set_title(title)
    if lower_limit:
        g.set_ylim(bottom=lower_limit)
    ax = g.axes
    ax.set_xscale('log')
    # for lh in g.legend_.legendHandles:
    #     lh.set_alpha(1)
    #     lh._sizes = [500]
    # g.legend_.markerscale = 1.0
    plt.legend(markerscale=2)
    if altitude_horizontal_line:
        plt.hlines(altitude_horizontal_line, 0, results[x].max())
    if path:
        fig = g.get_figure()
        fig.savefig(path)
    else:
        plt.show()


def runtime_ecdf(results, path):
    methods = results['method'].unique()
    for method in methods:
        hyperparams = results[results['method'] == method]['hyperparam'].unique()
        if len(hyperparams) == 1:
            g = sns.FacetGrid(data=results[results['method'] == method],
                              col='baseline_target_perf_ratio', row='nb_trajectories',
                              margin_titles=True)
            g.map(sns.ecdfplot, 'time')
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
            method = method.replace('<', '_smaller_')
            g.savefig(os.path.join(path, method))
        else:
            g = sns.FacetGrid(data=results[results['method'] == method],
                              col='baseline_target_perf_ratio', row='nb_trajectories', hue='hyperparam',
                              margin_titles=True)
            g.map(sns.ecdfplot, 'time')
            g.add_legend(title='Algorithm hyperparameter')
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

            method = method.replace('<', '_smaller_')
            g.savefig(os.path.join(path, method))


def heatmaps(results, path):
    # Heatmaps of all batch_rl_algorithms for all hyperparameters (mean)
    path_heatmaps = os.path.join(path, 'Intra batch_rl_algorithms comparisons',
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


def heatmaps_cm(results, path_results):
    # Heatmaps of all batch_rl_algorithms for all hyperparameters (Cvar)
    path_heatmaps = os.path.join(path_results, 'Random_mdps_malicious', 'Figures',
                                 'Intra batch_rl_algorithms comparisons',
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


def load_data_first_time(exp_path):
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
    return pd.concat(results_list)
