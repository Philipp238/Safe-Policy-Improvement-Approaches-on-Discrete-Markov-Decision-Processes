import os
import sys
import configparser
import time

import multiprocessing
import numpy as np

from experiment import WetChickenExperiment, RandomMDPsExperiment

directory = os.path.dirname(os.path.expanduser(__file__))
sys.path.append(directory)

path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'config.ini'))
results_directory_absolute = path_config['PATHS']['results_path']

config_name = sys.argv[1]
experiment_config = configparser.ConfigParser()
experiment_config.read(os.path.join(directory, 'experiments', config_name))
experiment_directory_relative = experiment_config['META']['experiment_path_relative']
environment = experiment_config['META']['env_name']
machine_specific_directory = sys.argv[2]

experiment_directory = os.path.join(results_directory_absolute, experiment_directory_relative)
machine_specific_experiment_directory = os.path.join(experiment_directory, machine_specific_directory)

if not os.path.isdir(results_directory_absolute):
    os.mkdir(results_directory_absolute)
if not os.path.isdir(experiment_directory):
    os.mkdir(experiment_directory)
if not os.path.isdir(machine_specific_experiment_directory):
    os.mkdir(machine_specific_experiment_directory)

nb_iterations = int(sys.argv[5])


def run_experiment(seed):
    if environment == 'wet_chicken':
        experiment = WetChickenExperiment(experiment_config=experiment_config, seed=seed, nb_iterations=nb_iterations,
                                          machine_specific_experiment_directory=machine_specific_experiment_directory)
    else:
        experiment = RandomMDPsExperiment(experiment_config=experiment_config, seed=seed, nb_iterations=nb_iterations,
                                          machine_specific_experiment_directory=machine_specific_experiment_directory)
    experiment.run()


def start_process():
    print('Starting', multiprocessing.current_process().name)


if __name__ == '__main__':
    pool_size = int(sys.argv[4])
    pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
    seed = int(sys.argv[3])
    ss = np.random.SeedSequence(seed)
    seeds = ss.generate_state(pool_size)

    f = open(os.path.join(machine_specific_experiment_directory, "Exp_description.txt"), "w+")
    f.write(
        f"This is the wet chicken benchmark with a heuristic baseline.\n")
    f.write(f'{pool_size} threads are being used, each computing {nb_iterations} iterations.\n')
    f.write(f'The seed which was used for the seed sequence is {seed} and the produced sequence is {seeds}.\n')
    f.write(f'Experiment starts at {time.ctime()}.')
    f.close()
    pool.map(run_experiment, seeds)
