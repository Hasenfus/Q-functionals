from DDPG import DDPGNet

import argparse
import os
from time import time
import pickle

import numpy as np
import torch
import gym
import sys
# from functional_critic import FourierBasis
from functional_critic import utils, utils_for_q_learning
# from functional_critic.FourierBasis import Net as FourierNet
# from functional_critic.PolynomialBasis import Net as PolynomialNet
# from functional_critic.LegendreBasis import Net as LegendreNet
from functional_critic.agents import FourierAgent, PolynomialAgent, LegendreAgent

from general.logging_utils import MetaLogger
# from datetime import datetime
from time import time
import json

from tqdm import tqdm

NUM_SAMPLES_TO_TRY = [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000]

# NUM_SAMPLES_TO_TRY = [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 15000, 20000, ]
# NUM_SAMPLES_TO_TRY = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, ]
# NUM_SAMPLES_TO_TRY = [1, 1000, 2000, 3000, 4000, 5000]
# NUM_SAMPLES_TO_TRY = [1, 1000, 2000]



def _get_device(device):
    if device == "cuda":
        return torch.device("cuda")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise Exception()

def _get_q_constructor(functional):
    if functional == "fourier":
        return FourierAgent
    elif functional == "polynomial":
        return PolynomialAgent
    elif functional == "legendre":
        return LegendreAgent
    elif functional == "ddpg":
        return DDPGNet
    else:
        raise Exception(f"Illegal functional: {functional}")

def make_q_functional_agent_and_env():
    hyper_parameter_directory = "./hyper_parameters"
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="./hyper_parameters",
                        type=str)
    parser.add_argument("--hyper_parameters_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name
    parser.add_argument("--device", default="cuda", type=str)

    # Need these so it eats it up and doesnt overwrite fields that dont exist
    parser.add_argument("--network_name", type=str, required=True)
    parser.add_argument("--filename", type=str, default="experiments/speed_test_results/raw/speed_tests.json")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--precomputed_speed_test", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}
    
    params = utils.get_hyper_parameters(args.hyper_parameters_name,
                                        hyper_parameter_directory)
    params['seed'] = 0

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    train_env = utils.make_env(params['env_name'], params.get("stop_flipping", False))

    device = _get_device(args.device)

    Q_Constructor = _get_q_constructor(params['functional'])

    train_env = utils.make_env(params['env_name'], params.get("stop_flipping", False))

    Q_object = Q_Constructor(
        params,
        train_env,
        state_size = train_env.observation_space.shape[0],
        action_size = train_env.action_space.shape[0],
        device=device,
        seed=params['seed'],
    )

    return Q_object, train_env


def test_q_functional_agent(Q_object, env):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--network_name", type=str, required=True)
    parser.add_argument("--filename", type=str, default="experiments/speed_test_results/speed_tests.json")
    parser.add_argument("--use_precomputed_basis", type=utils.boolify, default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    batch_size = args.batch_size
    device = Q_object.device

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    print(f"state_size: {state_size}, action_size: {action_size}")
    random_states = torch.rand(size=(batch_size, state_size)).to(device)

    speed_results = []
    for sample_size in NUM_SAMPLES_TO_TRY:
        random_actions = torch.rand(size=(batch_size, sample_size, action_size), device=device)
        torch.cuda.synchronize()
        start = time()
        try:
            for _ in range(100):
                if args.use_precomputed_basis:
                    Q_object.get_all_q_values_and_action_set(random_states, prefetch_basis_shape=(batch_size, sample_size))
                else:
                    Q_object.get_all_q_values_and_action_set(random_states, random_actions)
            torch.cuda.synchronize()
            duration_for_sample = time() - start
            print("Sampling", sample_size, "actions took", duration_for_sample)
            speed_results.append((sample_size, duration_for_sample))
        except:
            print(f"Broke down on {sample_size}")

    if not args.dry_run:
        import json
        try:
            with open(args.filename, "r") as f:
                current_speed_results = json.load(f)
        except:
            current_speed_results = {}

        current_speed_results[args.network_name] = speed_results
        new_json_str = json.dumps(current_speed_results, indent=4)
        print(new_json_str)

        with open(args.filename, "w") as f:
            f.write(new_json_str)


def do_speed_test_q_functional():
    Q_object, env = make_q_functional_agent_and_env()
    test_q_functional_agent(Q_object, env)


if __name__ == "__main__":
    do_speed_test_q_functional()
    print('big win')
    exit()
