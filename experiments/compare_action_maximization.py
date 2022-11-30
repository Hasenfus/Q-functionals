"""
Compare action maximization, for both SB3 and us. Involves:
    * loading a model from path
    * Getting a variety of states from the environment
    * Sampling a huge number of actions and evaluating them
    * Getting the "best" action outputted by the model's sampling strategy
    * Making some sort of comparison plot.

It might be best to do this for just one state. I'm not sure how to visualize it
for many. There's too many things to average over. Maybe in the caption we could say,
on average PG methods get the top 1% of actions, while ours gets the top 0.1%.

Example log:
ccv_results/path/to/logs/object_500000

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

from load_final_models import (load_final_ant_DDPG, 
                               load_final_ant_TD3, 
                               load_final_bipedal_DDPG, 
                               load_final_bipedal_TD3, 
                               load_final_hopper_DDPG, 
                               load_final_hopper_TD3, 
                               load_final_humanoid_DDPG, 
                               load_final_humanoid_TD3, 
                               load_final_lunar_lander_DDPG, 
                               load_final_lunar_lander_TD3, 
                               load_final_pendulum_DDPG, 
                               load_final_pendulum_TD3, 
                               load_final_reacher_DDPG, 
                               load_final_reacher_TD3, 
                               load_final_walker_DDPG, 
                               load_final_walker_TD3, 
                               )
from load_final_models import load_final_models_generic



def load_model_and_env_them_bad():
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

    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float)

    parser.add_argument("--entropy_scale",
                        default=None,
                        type=float)

    parser.add_argument("--load_model_path", type=str, required=True)

    args, unknown = parser.parse_known_args()


    train_env = utils.make_env(env_name=args.task)

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

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[256, 256], qf=[256, 256]))
    print("no action noise, not that it should make a difference")
    noise = NormalActionNoise(mean=0, sigma=0.0)
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
    print('loading model')
    thing = model.load(args.load_model_path)
    print('model loaded')
    import ipdb; ipdb.set_trace()
    return model, train_env

def load_model_and_env_them(model=None, task=None, load_model_path=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        # required=True,
                        default="DDPG",
                        type=str)

    parser.add_argument("--task",
                        # required=True,
                        type=str)

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


def get_states(model, env, save_file):
    """
    Save all states to file
    """
    states = []
    s = env.reset()
    done = False
    total_reward = 0
    while not done:
        states.append(s)
        s, r, done, _ = env.step(model.predict(s))
        total_reward += r
    print(f"Total reward for run: {total_reward}")
    if input("Save? (y/n)") == "y":
        np.save(save_file, states)
        print(f"Saved to {save_file}")
    else:
        print("Not saving")

def load_model_and_env_us():
    """
    Sort of annoying that I have to pass in a whole bunch of things to get this right.
    Params, etc. Maybe I just bite the bullet. I can do it in here.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameters_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name

    parser.add_argument("--seed", default=0,
                        type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--load_model_path", type=str, required=True)

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}
    
    params = utils.get_hyper_parameters(args.hyper_parameters_name,
                                        args.hyper_param_directory)

    params["hyper_parameters_name"] = args.hyper_parameters_name

    params['seed'] = args.seed

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    env = utils.make_env(params['env_name'], params.get("stop_flipping", False))
    params['env'] = env
    # utils_for_q_learning.set_random_seed(params)
    command_string = '"python ' + " ".join(sys.argv) + '"'
    params["command_string"] = command_string
    device = torch.device("cpu")

    Q_Constructor = None
    
    assert params['functional'] in ["fourier", "polynomial", "legendre"], "Functional type is not acceptable!"
    if params['functional'] == "fourier":
        Q_Constructor = FourierAgent
    elif params['functional'] == "polynomial":
        Q_Constructor = PolynomialAgent
    elif params['functional'] == "legendre":
        Q_Constructor = LegendreAgent
        
    Q_object = Q_Constructor(
        params,
        env,
        state_size = env.observation_space.shape[0],
        action_size = env.action_space.shape[0],
        device=device,
        seed=args.seed
    )
    Q_object.load_state_dict(torch.load(args.load_model_path, map_location=torch.device('cpu')))

    return Q_object, env

def get_many_actions(env, num_actions):
    action_dim = env.action_space.high.shape[0]
    max_a = env.action_space.high[0]
    all_actions = np.random.uniform(low=-max_a, high=max_a, size=(num_actions, action_dim))
    return all_actions

def get_q_values_from_many_actions_us(model, state, env, num_actions):
    all_actions = get_many_actions(env, num_actions)
    all_actions_torch = torch.FloatTensor(all_actions)
    state_torch = torch.FloatTensor(state)
    all_q_values_torch = model.get_all_q_values_and_action_set(state_torch, actions=all_actions_torch)
    all_q_values = all_q_values_torch.cpu().numpy().reshape(-1)
    return all_actions, all_q_values

def get_q_values_from_many_actions_them(model, state, env, num_actions):
    q_function = model.policy.critic.qf0
    all_actions = get_many_actions(env, num_actions)
    all_states = np.array([state for _ in range(num_actions)])
    state_actions = np.concatenate([all_states, all_actions], axis=1)
    state_actions_torch = torch.FloatTensor(state_actions)
    all_q_values = q_function(state_actions_torch).cpu().detach().numpy().reshape(-1)
    return all_actions, all_q_values
    pass

def get_best_action_and_q_value_us(model, state, env):
    all_actions, all_q_values = get_q_values_from_many_actions_us(model, state, env, 1000)
    max_q_index = np.argmax(all_q_values)
    max_q = all_q_values[max_q_index]
    max_action = all_actions[max_q_index]
    return max_action, max_q

def get_best_action_and_q_value_them(model, state):
    assert len(state.shape) == 1
    state_torch = torch.FloatTensor(state)[None,...]
    actor = model.policy.actor
    best_action = actor(state_torch).cpu().detach().numpy()

    best_action = model.predict(state)[0][None,...]

    q_function = model.policy.critic.qf0
    state_action = np.concatenate([state[None,...], best_action], axis=1)
    state_action_torch = torch.FloatTensor(state_action)
    q_value = q_function(state_action_torch).cpu().detach().numpy().item()
    assert len(best_action.shape) == 2
    assert best_action.shape[0] == 1
    best_action_single = best_action[0]
    return best_action_single, q_value


def main_us():
    num_actions_background = 100000
    model, env = load_model_and_env_us()
    state = env.reset()
    actions, q_values = get_q_values_from_many_actions_us(model, state, env, num_actions=num_actions_background)
    best_action, best_q_value = get_best_action_and_q_value_us(model, state, env)

    best_of_set, worst_of_set = q_values.max(), q_values.min()
    num_sampled_better = len([v for v in q_values if v > best_q_value])
    percent_better = num_sampled_better / len(q_values)
    print(f"Better than all but {percent_better:9.6f}")


def main_them():
    num_actions_background = 100000
    model, env = load_model_and_env_them()
    model.set_env(env)
    for _ in range(10):
        state = env.reset()
        actions, q_values = get_q_values_from_many_actions_them(model, state, env, num_actions=num_actions_background)
        best_action, best_q_value = get_best_action_and_q_value_them(model, state)
        best_of_set, worst_of_set = q_values.max(), q_values.min()
        num_sampled_better = len([v for v in q_values if v > best_q_value])
        percent_better = num_sampled_better / len(q_values)

        print(f"Worst: {worst_of_set:9.6f} Best: {best_of_set:9.6f} Policy: {best_q_value:9.6f} Better than all but {percent_better:9.6f}")


def make_placeholder_graph():
    import seaborn as sns; sns.set()
    x_data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_data = [0.0, 0.05, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    zeros = [0.0 for _ in range(len(x_data))]
    plt.rcParams['figure.figsize'] = (24, 16)
    plt.plot(x_data, y_data, linewidth=6, color='blue')
    plt.fill_between(x_data, y_data, zeros, color='blue', alpha=0.3)
    plt.hlines(0.5, 0, 1, colors='red', linewidth=6)
    plt.title("PLACEHOLDER CDF of policy quality over states", size=64)
    plt.xlabel("State", size=36)
    plt.ylabel("Action Quality", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.savefig("comparison_cdf.png", bbox_inches='tight')
    plt.close()


def get_ten_states_from_episode(model, env):
    state, done = env.reset(), False
    total_reward = 0
    all_states = []
    total_steps = 0
    while not done:
        total_steps += 1
        all_states.append(state)
        action = model.predict(state)[0]
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Total reward: {total_reward}")
    print(f"steps: {total_steps}")
    return [random.choice(all_states) for _ in range(10)]


def main_get_states_them(num_eps=10, write_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_10_eps"):
    model, env = load_model_and_env_them()
    model.set_env(env)

    all_states = []
    for num in range(num_eps):
        print(num)
        state = get_ten_states_from_episode(model, env)
        all_states.extend(state)

    all_states = np.array(all_states)
    np.save(write_path, all_states)
    print(f"Saved {len(all_states)} states to {write_path}")

def main_get_start_states_them(num_states=10, write_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_start_states"):
    model, env = load_model_and_env_them()
    model.set_env(env)

    all_states = []
    for num in range(num_states):
        state = env.reset()
        all_states.append(state)

    all_states = np.array(all_states)
    np.save(write_path, all_states)
    print(f"Saved {len(all_states)} states to {write_path}")

def get_read_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_path", type=str, default="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_10_eps")
    args = parser.parse_known_args()[0]
    return args.read_path

def get_write_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_path", type=str, default="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_10_eps")
    args = parser.parse_known_args()[0]
    return args.read_path

def main_evaluate_states_them(read_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_10_eps"):
    read_path = get_read_path()
    num_actions_background = 100000
    model, env = load_model_and_env_them()
    model.set_env(env)
    states = np.load(read_path)
    for state in states:
        actions, q_values = get_q_values_from_many_actions_them(model, state, env, num_actions=num_actions_background)
        best_action, best_q_value = get_best_action_and_q_value_them(model, state)
        best_of_set, worst_of_set = q_values.max(), q_values.min()
        num_sampled_better = len([v for v in q_values if v > best_q_value])
        percent_better = num_sampled_better / len(q_values)

        print(f"Worst: {worst_of_set:9.6f} Best: {best_of_set:9.6f} Policy: {best_q_value:9.6f} Better than all but {percent_better:9.6f}")

def test_model():
    model, env = load_final_pendulum_DDPG()
    model.set_env(env)
    for _ in range(10):
        state, done = env.reset(), False
        total_reward = 0
        while not done:
            state, r, done, _ = env.step(model.predict(state)[0])
            total_reward += r
        print(f"Total reward: {total_reward}")

def try_it_out():
    num_actions_background = 100000
    # model, env = load_final_bipedal_ddpg()
    # model, env = load_final_bipedal_TD3()
    # model, env = load_final_walker_ddpg()
    # model, env = load_final_bipedal_TD3()
    # model, env = load_final_humanoid_TD3()
    # model, env = load_final_pendulum_DDPG()
    model, env = load_final_pendulum_TD3()
    # model, env = load_final_reacher_DDPG()
    model.set_env(env)

    for _ in range(10):
        state = env.reset()
        actions, q_values = get_q_values_from_many_actions_them(model, state, env, num_actions=num_actions_background)
        best_action, best_q_value = get_best_action_and_q_value_them(model, state)
        best_of_set, worst_of_set = q_values.max(), q_values.min()
        num_sampled_better = len([v for v in q_values if v > best_q_value])
        percent_better = num_sampled_better / len(q_values)

        print(f"Worst: {worst_of_set:9.6f} Best: {best_of_set:9.6f} Policy: {best_q_value:9.6f} Better than all but {percent_better:9.6f}")

def fraction_less_then(l, n):
    return len([v for v in l if v < n]) / len(l)

def save_example_data_ours():
    """
    Luckily, we can just make this data ourselves. Not sure if we'll actually
    use it, but it's just easy math.
    """
    save_path = "./experiments/results_compare_action_maximization/Ant-v3/data/q_functional_cdf_data.npy"
    # num_actions_background = 1000
    num_actions_sampled = 1000
    total_steps = 1000
    all_fractions = []
    for step in range(total_steps):
        best_value_sampled = max([random.random() for _ in range(num_actions_sampled)])
        # It's pretty easy to see what this is better than. all but 1 - value things.
        fraction_better = 1 - best_value_sampled
        print(fraction_better)
        all_fractions.append(fraction_better)
    
    print("sorting fractions")
    all_fractions.sort()
    print(f"Saving {len(all_fractions)} numbers")
    all_fractions = np.array(all_fractions)
    np.save(save_path, all_fractions)
    print("saved")

def save_data_ant_ddpg():
    save_path = "./experiments/results_compare_action_maximization/Ant-v3/data/DDPG_cdf_data.npy"
    steps_per_seed = 1000
    num_actions_background = 1000
    seeds = [101, 102, 103]
    all_fractions = []
    for seed in seeds:
        print(f"Seed {seed}")
        model, env = load_final_ant_DDPG(seed=seed)
        assert env.action_space.low[0] == -1.0
        model.set_env(env)
        state, done = env.reset(), False
        steps = 0
        while steps < steps_per_seed:
            if steps % 100 == 0:
                print(f"step {steps}")
            actions, q_values = get_q_values_from_many_actions_them(model, state, env, num_actions=num_actions_background)
            best_action, best_q_value = get_best_action_and_q_value_them(model, state)
            num_sampled_better = len([v for v in q_values if v > best_q_value])
            fraction_better = num_sampled_better / len(q_values)
            all_fractions.append(fraction_better)

            noisy_best_action = best_action + np.random.normal(0, 0.1, size=best_action.shape)
            noisy_best_action = np.clip(noisy_best_action, -1, 1)

            state, r, done, _ = env.step(noisy_best_action)

            steps += 1
            if done:
                print("done, resetting env")
                state, done = env.reset(), False


    print("sorting fractions")
    all_fractions.sort()
    print(f"Saving {len(all_fractions)} numbers")
    all_fractions = np.array(all_fractions)
    np.save(save_path, all_fractions)
    print("saved")


def plot_ant_stuff():
    ddpg_data_path = "./experiments/results_compare_action_maximization/Ant-v3/data/DDPG_cdf_data.npy"
    sampling_data_path = "./experiments/results_compare_action_maximization/Ant-v3/data/q_functional_cdf_data.npy"

    ddpg_data = np.load(ddpg_data_path).tolist()
    sampling_data = np.load(sampling_data_path).tolist()

    y_axis_ddpg = np.linspace(0, 1, len(ddpg_data)).tolist()
    y_axis_ddpg.insert(0, 0.)
    y_axis_ddpg.append(1.)
    ddpg_data.insert(0, 0.)
    ddpg_data.append(1.)

    y_axis_sampling = np.linspace(0, 1, len(sampling_data)).tolist()
    y_axis_sampling.insert(0, 0.)
    y_axis_sampling.append(1.)
    sampling_data.insert(0, 0.)
    sampling_data.append(1.)

    plt.plot(ddpg_data, y_axis_ddpg, label="DDPG Policy CDF")
    plt.plot(sampling_data, y_axis_sampling, label="Sampling Policy CDF")
    plt.legend()
    plt.show()


def try_it_out_episode():
    seed=101
    # seed=102
    # seed=103
    # num_actions_background = 100000
    # num_actions_background = 1000
    num_actions_background = 2000
    model, env = load_final_ant_DDPG(seed=seed)
    # model, env = load_final_ant_TD3(seed=seed)
    # model, env = load_final_bipedal_DDPG(seed=seed)
    # model, env = load_final_bipedal_TD3(seed=seed)
    # model, env = load_final_hopper_DDPG(seed=seed)
    # model, env = load_final_hopper_TD3(seed=seed)
    # model, env = load_final_humanoid_DDPG(seed=seed)
    # model, env = load_final_humanoid_TD3(seed=seed)
    # model, env = load_final_lunar_lander_DDPG(seed=seed)
    # model, env = load_final_lunar_lander_TD3(seed=seed)
    # model, env = load_final_pendulum_DDPG(seed=seed)
    # model, env = load_final_pendulum_TD3(seed=seed)
    # model, env = load_final_reacher_DDPG(seed=seed)
    # model, env = load_final_reacher_TD3(seed=seed)
    # model, env = load_final_walker_DDPG(seed=seed)
    # model, env = load_final_walker_TD3(seed=seed)

    model.set_env(env)

    state, done = env.reset(), False
    total_reward = 0
    percent_list = []
    while not done:
        actions, q_values = get_q_values_from_many_actions_them(model, state, env, num_actions=num_actions_background)
        best_action, best_q_value = get_best_action_and_q_value_them(model, state)
        best_of_set, worst_of_set = q_values.max(), q_values.min()
        num_sampled_better = len([v for v in q_values if v > best_q_value])
        percent_better = num_sampled_better / len(q_values)
        percent_list.append(percent_better)

        print(f"Worst: {worst_of_set:9.6f} Best: {best_of_set:9.6f} Policy: {best_q_value:9.6f} Better than all but {percent_better:9.6f}")

        noisy_best_action = best_action + np.random.normal(0, 0.1, size=best_action.shape)
        assert env.action_space.low[0] == -1.0
        noisy_best_action = np.clip(noisy_best_action, -1, 1)
        state, r, done, _ = env.step(noisy_best_action)
        total_reward += r
    print(f"total reward: {total_reward}")

    # confusingly, the percents are the x
    sorted_percent_list = list(sorted(percent_list))
    y_axis = np.linspace(0, 1, len(sorted_percent_list)).tolist()
    sorted_percent_list.insert(0, 0.)
    sorted_percent_list.append(1.)
    y_axis.insert(0, 0.)
    y_axis.append(1.)
    print(sorted_percent_list)
    print("Fraction better than 0.01", fraction_less_then(sorted_percent_list, 0.01))
    print("Fraction better than 0.05", fraction_less_then(sorted_percent_list, 0.05))
    print("Fraction better than 0.1", fraction_less_then(sorted_percent_list, 0.1))
    print("Fraction better than 0.2", fraction_less_then(sorted_percent_list, 0.2))
    print("Fraction better than 0.5", fraction_less_then(sorted_percent_list, 0.5))
    print("Fraction better than 0.9", fraction_less_then(sorted_percent_list, 0.9))

    plt.plot(sorted_percent_list, y_axis)
    plt.show()




if __name__ == "__main__":
    # try_out_ddpg_bipedal()
    # try_out_TD3_bipedal()
    # try_it_out()
    # try_it_out_episode()
    # test_model()
    # main_us()
    # main_them()
    # make_placeholder_graph()
    # main_get_states_them(num_eps=10, write_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_100_eps_halfway")
    # main_evaluate_states_them(read_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_100_eps_halfway.npy")
    # main_get_start_states_them(num_states=100, write_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_100_eps_halfway_start")
    # main_evaluate_states_them(read_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_100_eps_halfway_start.npy")
    # main_evaluate_states_them(read_path="./experiments/results_compare_action_maximization/HalfCheetah/states/1000_states_100_eps_halfway.npy")
    # save_data_ant_ddpg()
    # save_example_data_ours()
    plot_ant_stuff()