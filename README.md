## Prerequisites

This repository contains the code accompanying the Master's thesis "Evaluation of Safe Policy 
Improvement by Soft Baseline Bootstrapping" of Philipp Scholl (Technical University of Munich), which 
investigates safe reinforcement learning
by building on the paper "Safe Policy Improvement with Soft Baseline Bootstrapping" by Nadjahi 
et al. [[1]](#1).

The code is implemented in Python 3 and requires the packages specified in ``requirements.txt``.

## Structure

The `batch_rl_algorithms/` folder is the main work of this repository containing every batch RL algorithm
considered in this thesis. The abstract base class is written in `batch_rl_algorithms/batch_rl_algorithm.py` 
and the actual algorithms are distributed over the whole folder. They should be used by initializing them
with a batch of data, a behavior policy used to collect this data and the corresponding hyper-parameters.
Additionally, it is necessary to provide basic information about the MDP, e.g. the number of actions and states and
whether is is an episodic MDP. During the initialization, the algorithm will calculate every quantity it needs
in the actual training phase, for example, the estimates of the transition probabilities. The training is started
by calling the `fit()` method on the algorithm object. You can find an example below:

````
adv_approx_soft_spibb = AdvApproxSoftSPIBB(pi_b, gamma, nb_states, nb_actions, data, 
                                           episodic, R, delta, epsilon, 'hoeffding')
adv_approx_soft_spibb.fit()
````

`wet_chicken_discrete/` contains the Wet Chicken benchmark one of the two Benchmarks used in the Master's thesis 
"Evaluation of Safe Policy Improvement by Soft Baseline Bootstrapping". The file `wet_chicken_discrete/dynamics.py`
implements the dynamics of this benchmark and `wet_chicken_discrete/baseline_policy.py` implements different methods
to compute a behavior policy. In my thesis, I only make use of the `'heuristic'` variant.



`auxiliary_tests/` contains 


## Commands

To generate a dataset for the helicopter environment, run:

`python baseline.py baseline/helicopter_env/ weights.pt --generate_dataset --dataset_size 1000`

where ``baseline/helicopter_env/`` is the path to the baseline and ``weights.pt`` is the baseline filename. The dataset will be generated in ``baseline/helicopter_env/dataset`` by default. We have provided the baseline used in our experiment in `baseline/helicopter_env`. To train a new one, define the training parameters in a yaml file e.g. `config` and run:

`ipython train.py -- --domain helicopter --config config`

To train a policy on that dataset, define the training parameters in a yaml file e.g. `config_batch` (in particular, that file should contain the path to the baseline and dataset to use) and then run:

`ipython train.py -- -o batch True --config config_batch`

To specify different learning types or parameters, either change the `config_batch` file or pass options to the command line, e.g. `--options learning_type ramdp`, or `--options minimum_count 5` (see the file `config_batch.yaml` for the existing options). In particular, the learning_type parameter can be regular (ie DQN), pi_b (ie SPIBB-DQN), soft_sort (ie Soft-SPIBB-DQN) or ramdp.

Results for the run will be saved in the dataset folder, in a csv file named `{algo_runid_parameter}`.

We also provide a baseline for CartPole, use the following commands to train on that environment:

`python baseline.py baseline/cartpole_env/ weights.pt --generate_dataset --dataset_size 1000`

`ipython train.py -- -o batch True --config config_batch_cartpole --domain gym`.


## References

[1] K. Nadjahi, R. Laroche, R. Tachet des Combes. *Safe
			Policy Improvement with Soft Baseline Bootstrapping*. Proceedings of the 2019
		European Conference on Machine Learning and Principles and Practice of Knowledge
		Discovery in Databases (ECML-PKDD). 2019.
		
[2] Alexander Hans and Steffen Udluft. *Efficient
			Uncertainty Propagation for Reinforcement Learning with Limited Data*.
		International Conference on Artificial Neural Networks. Springer. 2009. pp.
		70–79.