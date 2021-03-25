## Prerequisites

This repository contains the code accompanying the Master's thesis "Evaluation of Safe Policy 
Improvement by Soft Baseline Bootstrapping" of Philipp Scholl (Technical University of Munich), which 
investigates safe reinforcement learning
by building on the paper "Safe Policy Improvement with Soft Baseline Bootstrapping" by Nadjahi 
et al. [1] and their code https://github.com/RomainLaroche/SPIBB.

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
The `PiStar` class implements dynamic programming and should be used with the true MDP dynamics to compute the optimal
policy.

`wet_chicken_discrete/` contains the Wet Chicken benchmark [2], which is one of the two benchmarks used in 
"Evaluation of Safe Policy Improvement by Soft Baseline Bootstrapping". The file `wet_chicken_discrete/dynamics.py`
implements the dynamics of this benchmark and `wet_chicken_discrete/baseline_policy.py` implements different methods
to compute a behavior policy. In my thesis, I only make use of the `'heuristic'` variant. 

To run experiments using the algorithms implemented in `batch_rl_algorithms/` run:

``run_experiments.py wet_chicken_full.ini wet_chicken_results 1234 4 10``

The `wet_chicken_full.ini` is the name of the config file used for this experiment, more about this later. 
`wet_chicken_results` is the folder name, where the results are going to be stored, `1234` is the seed for the 
experiment, `4` is the number of threads and `10` is the number of iterations performed per thread per algorithm.
The previous mentioned config file has to be stored in the folder `experiments/` and contains parameters about:

1. the experiment itself (storage path, which benchmark, speedup function etc.),
2. the environment parameters,
3. the behavior/baseline policy parameters and
4. the algorithms and their hyper-parameters.

These experiments can be conducted either on the Wet Chicken benchmark or on the Random MDPs benchmark [1]. An example
config file for the Random MDPs is given as `experiments/random_mdps_full.ini`. Before running the experiments, you
have to create a file called `paths.ini` (on the highest directory level) which contains the following:
````
[PATHS]
results_path = D:\results
spibb_path = C:\users\dummy\SPIBB
````
Where `results_path` should be the absolute path pointing to the place where the results should be stored (I store
the results outside of this repository as the results are often huge (>1GB)). `spibb_path` is only necessary if you use
anything from https://github.com/RomainLaroche/SPIBB, which is the case if you choose the Random MDPs benchmark in your
experiments or some of the tests in `auxiliary_tests/`. The `spibb_path` has to be the absolute path to a local copy
of https://github.com/RomainLaroche/SPIBB.

`auxiliary_tests/` contains complementary code for some tests described in "Evaluation of Safe Policy Improvement using 
Soft Baseline Bootstrapping".


## References

[1] K. Nadjahi, R. Laroche, R. Tachet des Combes. *Safe
			Policy Improvement with Soft Baseline Bootstrapping*. Proceedings of the 2019
		European Conference on Machine Learning and Principles and Practice of Knowledge
		Discovery in Databases (ECML-PKDD). 2019.
		
[2] Alexander Hans and Steffen Udluft. *Efficient
			Uncertainty Propagation for Reinforcement Learning with Limited Data*.
		International Conference on Artificial Neural Networks. Springer. 2009. pp.
		70–79.