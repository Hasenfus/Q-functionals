"""
This was getting out of hand in the other file.
"""

import sys, os
import numpy as np
import gym
import random
import argparse
from functional_critic import utils, utils_for_q_learning
from functional_critic.agents import FourierAgent, LegendreAgent, PolynomialAgent
import torch

from gym.wrappers import TransformReward
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer

import matplotlib.pyplot as plt



def load_model_and_env_them(model=None, task=None, load_model_path=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        # required=True,
                        default="DDPG",
                        type=str)

    parser.add_argument("--task",
                        # required=True,
                        type=str)

    # parser.add_argument("--experiment_name",
    #                     required=True,
    #                     type=str)

    # parser.add_argument("--run_title",
    #                     required=True,
    #                     type=str)

    parser.add_argument("--load_model_path", type=str)#, required=True)

    args, unknown = parser.parse_known_args()

    if model:
        args.model = model
    if task:
        args.task = task
    if load_model_path:
        args.load_model_path = load_model_path

    train_env = utils.make_env(env_name=args.task)

    if args.model == "DDPG":
        constructor = DDPG
    elif args.model == "SAC":
        constructor = SAC
    elif args.model == "TD3":
        constructor = TD3

    model = constructor.load(args.load_model_path)

    return model, train_env


def load_final_models_generic(seed, model_name, task):
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env



def load_final_ant_DDPG(seed=101):
    model_name = "DDPG"
    task = "Ant-v3"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env


def load_final_ant_TD3(seed=101):
    model_name = "TD3"
    task = "Ant-v3"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    print(model_path)
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env


def load_final_bipedal_DDPG(seed=101):
    model_name = "DDPG"
    task = "BipedalWalker-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_bipedal_TD3(seed=101):
    model_name = "TD3"
    task = "BipedalWalker-v3"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_hopper_DDPG(seed=101):
    model_name = "DDPG"
    task = "Hopper-v3"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_hopper_TD3(seed=101):
    model_name = "TD3"
    task = "Hopper-v3"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_humanoid_DDPG(seed=101):
    model_name = "DDPG"
    task = "Humanoid-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_humanoid_TD3(seed=101):
    model_name = "TD3"
    task = "Humanoid-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_lunar_lander_DDPG(seed=101):
    model_name = "DDPG"
    task = "LunarLanderContinuous-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_lunar_lander_TD3(seed=101):
    model_name = "TD3"
    task = "LunarLanderContinuous-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_pendulum_DDPG(seed=101):
    model_name = "DDPG"
    task = "Pendulum-v1"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_pendulum_TD3(seed=101):
    model_name = "TD3"
    task = "Pendulum-v1"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_reacher_DDPG(seed=101):
    model_name = "DDPG"
    task = "Reacher-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/DDPG/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/DDPG_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_reacher_TD3(seed=101):
    model_name = "TD3"
    task = "Reacher-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_walker_DDPG(seed=101):
    model_name = "DDPG"
    task = "Walker2d-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/DDPG/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/DDPG_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

def load_final_walker_TD3(seed=101):
    model_name = "TD3"
    task = "Walker2d-v2"
    model_path = f"ccv_sb3_results/sb3_results/all_gym_envs/full_runs/{model_name}/more_seeds_1/stable_baselines_{model_name}_gym_full_three_more_seeds__task_{task}/seed_{seed}/model/{model_name}_{task}_seed_{seed}.zip"
    model, env = load_model_and_env_them(
        model=model_name, task=task, load_model_path=model_path)

    return model, env

