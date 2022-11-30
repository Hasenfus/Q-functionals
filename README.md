# Accompanying code for "Q-Functionals for Value-Based Continuous Control" 

### Installation
1. Download and extract code
1. Create a virtualenv with `python3 -m venv venv`
2. Activate venv with `source ./venv/bin/activate`
3. Install dependencies with `pip install -r requirements`



### Running code
Below is an example of launching a single job on the Pendulum task:

```
python experiments/experiment.py --experiment_name results/pendulum/example_runs --run_title first_run --functional legendre --rank 3 --use_quantile_sampling_bootstrapping False --hyper_parameters_name gym --env_name Pendulum-v0
```

We use the library [onager](https://github.com/camall3n/onager) for large sets of experiments with one command. To launch the entire benchmark suite of experiments for Q-Functional (DDPG analogue), first run the command

```
onager prelaunch +jobname q_functional_ddpg_all_envs_all_seeds +command "python experiments/experiment.py --experiment_name results/gym_all_envs/full_runs/q_functional_ddpg/first --hyper_parameters_name gym --use_precomputed_basis True" +arg --env_name Pendulum-v1 Reacher-v2 LunarLanderContinuous-v2 BipedalWalker-v3 Hopper-v3 Walker2d-v2 Ant-v3 Humanoid-v2 +arg --seed 0 1 2 3 4 5 6 7
```

This creates configurations for 8 seeds of all 8 environments. To launch these jobs, then run:

```
onager launch --jobname q_functional_ddpg_all_envs_all_seeds --backend local --max-tasks 2
```

To launch the entire benchmark suite of experiments for Q-functional (TD3 analogue), first run the command

```
onager prelaunch +jobname q_functional_td3_all_envs_all_seeds +command "python experiments/experiment.py --experiment_name results/gym_all_envs/full_runs/q_functional_td3/first --hyper_parameters_name gym --use_precomputed_basis True --use_quantile_sampling_bootstrapping True --minq True" +arg --env_name Pendulum-v1 Reacher-v2 LunarLanderContinuous-v2 BipedalWalker-v3 Hopper-v3 Walker2d-v2 Ant-v3 Humanoid-v2 +arg --seed 0 1 2 3 4 5 6 7
```

To then launch these jobs, run:

```
onager launch --jobname q_functional_td3_all_envs_all_seeds --backend local --max-tasks 2
```


The entrypoint for running the Q-functional algorithm is `experiments/experiment.py`, and the entrypoint for running policy gradient baselines is `experiments/stable_baselines_experiments.py`. Plotting code for visualizing graphs can be found in `experiments/plot_learning_curves.py` and `experiments/sb3_plot_learning_curves.py`. Code for generating plots besides benchmarks can be found in
1. `experiments/bandit.py`
2. `experiments/compare_action_maximization.py`
3. `experiments/generate_speed_comparison.py` and `experiments/make_speed_plots.py`
4. `sampling_comparison.py`



##### Bibtext

```
@inproceedings{he2023qfunctional,
  title={Q-Functionals for Value-Based Continuous Control},
  author={He, Bowen and Lobel, Sam and Rammohan, Sreehari and Yu, Shangqun and Konidaris, George},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```