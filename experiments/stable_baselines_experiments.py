from ctypes import util
import gym
from gym.wrappers import TransformReward
import numpy as np
import torch
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from functional_critic import utils
import argparse
import os

"""
This file is used for doing baseline experiments with SB3, on DDPG, SAC and TD3.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--model",
                    required=True,
                    default="DDPG",
                    type=str)

parser.add_argument("--seed",
                    required=True,
                    default=0,
                    type=int)

parser.add_argument("--task",
                    required=True,
                    type=str)

parser.add_argument("--experiment_name",
                    required=True,
                    type=str)

parser.add_argument("--run_title",
                    required=True,
                    type=str)

parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float)

parser.add_argument("--entropy_scale",
                    default=None,
                    type=float)

parser.add_argument("--noise_std",
                    default=0.1,
                    type=float)

parser.add_argument("--timesteps",
                    default=None,
                    type=int)

args, unknown = parser.parse_known_args()
print("Training for Seed={}".format(args.seed))
log_dir = os.path.join(args.experiment_name, args.run_title, f"seed_{args.seed}")
logger = configure(folder=log_dir)
model_path = os.path.join(log_dir, "model", args.model+"_"+args.task+"_seed_"+str(args.seed))

train_env = gym.make(env_name=args.task)

clip_dict = {"BipedalWalker-v3":20,
         "Hopper-v3":50,
         "HalfCheetah-v3":20,
         "Ant-v3":20,
         "Reacher-v2":20,
         "Walker2d-v2":20,
         "Humanoid-v2":20}

buffer_dict = {"BipedalWalker-v3":200000,
         "Hopper-v3":200000,
         "HalfCheetah-v3":500000,
         "Ant-v3":500000,
         "Reacher-v2":200000,
         "Walker2d-v2":500000,
         "Humanoid-v2":250000}

timestep_dict = {"BipedalWalker-v3":1000000,
         "Hopper-v3":1000000,
         "HalfCheetah-v3":2000000,
         "Ant-v3":2000000,
         "Reacher-v2":200000,
         "Walker2d-v2":2000000,
         "Humanoid-v2":2000000}


# Defaults
reward_clip = clip_dict.get(args.task, 20)
buffer_size = buffer_dict.get(args.task, 500000)

timesteps_default = timestep_dict.get(args.task, 1000000)
timesteps = args.timesteps if args.timesteps is not None else timesteps_default

train_env = TransformReward(train_env, lambda x:np.clip(x, a_min=-reward_clip,
                a_max=reward_clip))
test_env = gym.make(env_name=args.task)

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[256, 256], qf=[256, 256]))

noise = NormalActionNoise(mean=0, sigma=args.noise_std)

assert args.model in ["DDPG", "SAC", "TD3"]
if args.model == "DDPG":
    constructor = DDPG
elif args.model == "SAC":
    constructor = SAC
elif args.model == "TD3":
    constructor = TD3

if args.entropy_scale is not None:
    model_kwargs = {'ent_coef': args.entropy_scale}
else:
    model_kwargs = {}

model = constructor("MlpPolicy",
                    train_env,
                    buffer_size=buffer_size,
                    learning_starts=10000,
                    batch_size=512,
                    learning_rate=args.learning_rate,
                    tau=0.005,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    action_noise=noise,
                    replay_buffer_class = ReplayBuffer,
                    seed=args.seed,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    **model_kwargs,
                    )
model.set_logger(logger)

model.learn(total_timesteps=timesteps, eval_env=test_env, eval_freq=10000, n_eval_episodes=5)
model.save(model_path)
print("Completed")