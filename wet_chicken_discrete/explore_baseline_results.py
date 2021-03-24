import pandas as pd
import numpy as np
import string
import re
import ast
import seaborn as sns


def str2array(s):
    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


### Fix learning_rate and epsilon
results = pd.read_csv(r'C:\Users\phili\PycharmProjects\safe_rl\wet_chicken_discrete\Results\testing_baseline.csv')

g = sns.relplot(data=results, x='max_nb_it', y='perf_baseline', col='learning_rate', row='epsilon', kind='line')
for ax in g.axes.flat:
    ax.set_xscale('log')

### Variable learning rate and epsilon
results = pd.read_csv(
    r'C:\Users\phili\PycharmProjects\safe_rl\wet_chicken_discrete\Results\testing_baseline_variable_learning.csv')
g = sns.relplot(data=results, x='max_nb_it', y='perf_baseline', col='order_learning_rate', row='order_epsilon',
                kind='line')
for ax in g.axes.flat:
    ax.set_xscale('log')

results_order_inf = pd.read_csv(
    r'C:\Users\phili\PycharmProjects\safe_rl\wet_chicken_discrete\Results\testing_baseline_variable_learning_epsilon_order_inf.csv')
g = sns.relplot(data=results, x='max_nb_it', y='perf_baseline', col='order_learning_rate', row='order_epsilon',
                kind='line')
for ax in g.axes.flat:
    ax.set_xscale('log')

results_all = pd.concat([results, results_order_inf])

g = sns.relplot(data=results_all, x='max_nb_it', y='perf_baseline', row='order_learning_rate', col='order_epsilon',
                kind='line')
for ax in g.axes.flat:
    ax.set_xscale('log')

results_all.to_csv(
    r'C:\Users\phili\PycharmProjects\safe_rl\wet_chicken_discrete\Results\testing_baseline_variable_learning.csv')

### state_count_dependent_variable
results = pd.read_csv(
    r'C:\Users\phili\PycharmProjects\safe_rl\wet_chicken_discrete\Results\testing_baseline_state_count_dependent_variable.csv')
g = sns.relplot(data=results, x='max_nb_it', y='perf_baseline', col='order_learning_rate', kind='line')
for ax in g.axes.flat:
    ax.set_xscale('log')
