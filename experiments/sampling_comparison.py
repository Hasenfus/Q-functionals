import argparse
import os
from time import time

import numpy as np
import torch
import gym
import sys
sys.path.append("..")
from functional_critic import utils, utils_for_q_learning
from functional_critic.agents import FourierAgent, PolynomialAgent, LegendreAgent
# from datetime import datetime
from time import time
from functional_critic.utils import *
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

"""
Compare returns when sampling different numbers of actions
"""

# seed 1 model file
s_1_model_file = "results/aaai_results2/humanoid/functional_TD3_dmc_humanoid_tanh_1__quantilesamplingpercent_0.01__finallayerinitscale_1.0__activation_tanh__learningrate_0.0001/logs/seed_1_object_1000000"
# seed 2 model file
s_2_model_file = "results/aaai_results2/humanoid/functional_TD3_dmc_humanoid_tanh_1__quantilesamplingpercent_0.01__finallayerinitscale_1.0__activation_tanh__learningrate_0.0001/logs/seed_2_object_1000000"
# filepath for all model hypers
filepath_hyper = 'results/aaai_results2/humanoid/functional_TD3_dmc_humanoid_tanh_1__quantilesamplingpercent_0.01__finallayerinitscale_1.0__activation_tanh__learningrate_0.0001/hyperparams/90__seed_1.hyper'

actions_to_try_sampling = [1, 2, 10, 50, 100, 300, 500, 700, 800, 1000, 1500, 2000, 3000, 4000, 5000]
# actions_to_try_sampling = [1, 2, 10]

env = gym.make("Humanoid-v2")

def process_all_results(json_filename, params):
    write_results_to_file(1, s_1_model_file, json_filename, params)
    write_results_to_file(2, s_2_model_file, json_filename, params)

def write_results_to_file(seed_num, model_file, json_file, params, n_evals=10):   
    try:
        with open(json_file, "r") as f:
            current_speed_results = json.load(f)
    except:
        current_speed_results = {}

    batch_size = 512

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    results = []
    std_devs = []

    for sample_size in actions_to_try_sampling:
        # create Legendre agent 

        params['qstar_samples'] = sample_size
         
        Q_object = LegendreAgent(
            params,
            env,
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            device=device,
            seed=params["seed"]
        )

        print("actions to sample", Q_object.action_to_sample)

        # load in weights
        Q_object.load_state_dict(torch.load(model_file))
        Q_object.eval()

        start_time = time()
        evaluation_rewards = []
        total_steps = 0
        for _ in range(n_evals):
            evaluation_reward = 0
            s_eval, done_eval = env.reset(), False
            while not done_eval:
                total_steps += 1
                a_eval = Q_object.e_greedy_policy(s_eval, -1, -1, 'test')
                s_eval_, r_eval, done_eval, _ = env.step(a_eval)
                evaluation_reward += r_eval
                s_eval = s_eval_
            evaluation_rewards.append(evaluation_reward)
        
        results.append(np.mean(evaluation_rewards))
        std_devs.append(np.std(evaluation_rewards))

        print(f"{sample_size} samples --> Evaluation Reward: {np.mean(evaluation_rewards)}")
        print(f"{sample_size} Took {time()-start_time:9.4f}")
        print(f"That's {(time()-start_time)/total_steps:9.7f} per step")
        del Q_object
        torch.cuda.empty_cache()

    results = np.array(results)
    std_errs = np.array(std_devs) / np.sqrt(n_evals)

    json_obj_for_seed = {"results": results.tolist(), "std_errs": std_errs.tolist()}

    # write the data to a file
    current_speed_results['seed_' + str(seed_num)] = json_obj_for_seed
    new_json_str = json.dumps(current_speed_results, indent=4)
    print(new_json_str)

    with open(args.filename, "w") as f:
        f.write(new_json_str)

def make_plot_from_json(filepath):
    try:
        with open(filepath, "r") as f:
            current_speed_results = json.load(f)
    except:
        current_speed_results = {}
        raise Exception("uh oh there should be some results for us to plot")

    num_seeds = [1, 2]
    data = []
    std_errs = []

    for seed in num_seeds:
        d = current_speed_results['seed_'+str(seed)]['results']
        e = current_speed_results['seed_'+str(seed)]['std_errs']
        d = np.array(d)
        e = np.array(e)
        data.append(d)
        std_errs.append(e)

    data = np.array(data)
    avg_data = data.mean(axis=0)
    std_errs = np.std(data, axis=0)/np.sqrt(2) #np.array(std_errs).mean(axis=0)
    import seaborn as sns; sns.set()
    plt.plot(actions_to_try_sampling, avg_data, color='royalblue')
    plt.fill_between(actions_to_try_sampling, avg_data-std_errs, avg_data+std_errs, alpha=0.2, color='royalblue')
    plt.title("Return vs. Num Actions Sampled")
    plt.xlabel("Num Actions Sampled")
    plt.ylabel("Return")
    plt.savefig("./experiments/sampling_comparison.png",  bbox_inches='tight')
    

if __name__ == "__main__":
    
    params = {} 

    # load in the params
    with open(filepath_hyper) as f:
        lines = [line.rstrip('\n') for line in f]
        for l in lines:
            parameter_name, parameter_value, parameter_type = (l.split(','))
            if parameter_type == 'string':
                params[parameter_name] = str(parameter_value)
            elif parameter_type == 'integer':
                params[parameter_name] = int(parameter_value)
            elif parameter_type == 'float':
                params[parameter_name] = float(parameter_value)
            elif parameter_type == 'boolean':
                params[parameter_name] = boolify(parameter_value)
            else:
                print("unknown parameter type ... aborting")
                print(l)
                sys.exit(1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", default=None, type=int, required=True) # set the rank for onager experiments.
    parser.add_argument("--functional", type=str, required=True)
        
    parser.add_argument("--filename", type=str, default="experiments/sampling_test.json")
    parser.add_argument("--dry_run", type=utils.boolify, default=False)
    parser.add_argument("--use_precomputed_basis", type=utils.boolify, default=False)
    parser.add_argument("--n_evals", type=int, default=10)

    args, unknown = parser.parse_known_args()

    assert args.functional in ["fourier", "polynomial", "legendre"], args.functional

    params["rank"] = args.rank
    params["use_precomputed_basis"] = args.use_precomputed_basis

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")


    make_plot_from_json(args.filename)
            
    