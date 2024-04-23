import argparse
import os
import statistics
from time import time
import pickle
from matplotlib.legend import Legend

import numpy as np
import torch
import gym
import sys
from functional_critic import utils, utils_for_q_learning
from functional_critic.agents import FourierAgent, LegendreAgent, PolynomialAgent
from general.logging_utils import MetaLogger


"""
This file is the main entry for experiments on MuJoCo tasks. It reads in arguments and runs experiments defined in hyperparameter files.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameters_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name

    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
                        required=True)

    parser.add_argument("--run_title",
                        type=str,
                        help="subdirectory for this run",
                        required=True)

    parser.add_argument("--seed", default=0,
                        type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--evaluation_frequency", default=10000,
                        required=False, type=int)

    parser.add_argument("--save_model", action="store_true")
    
    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}
    
    params = utils.get_hyper_parameters(args.hyper_parameters_name,
                                        args.hyper_param_directory)
    
    full_experiment_name = os.path.join(args.experiment_name, args.run_title)
    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))
    params["hyperparams_dir"] = hyperparams_dir
    params["hyper_parameters_name"] = args.hyper_parameters_name
    
    """
    The logging utils
    """
    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("average_loss", logging_filename)
    meta_logger.add_field("average_q", logging_filename)
    meta_logger.add_field("average_q_star", logging_filename)
    meta_logger.add_field("episodic_rewards", logging_filename)
    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("all_times", logging_filename)
    meta_logger.add_field("episodes_so_far", logging_filename)
    
    """
    The parameters controlling the environment
    """
    params['seed'] = args.seed

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    train_env = gym.make(params["env_name"])
    eval_env = gym.make(params["env_name"])

    params['env'] = train_env
    # utils_for_q_learning.set_random_seed(params)
    
    command_string = '"python ' + " ".join(sys.argv) + '"'
    params["command_string"] = command_string

    utils.save_hyper_parameters(params, args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("running on CUDA...")
    else:
        device = torch.device("cpu")
        print("running on CPU...")
        
        
    """
    Print out the configurations.
    """
    print("{:=^100s}".format("Basic Configurations"))
    print("Training Environment:", params["env_name"])
    print("Functional:", params['functional'])
    print("rank:", params['rank'])
    print("seed:", params['seed'])
    
    print("{:=^100s}".format("Model Configurations"))
    print("TD3 Trick:", params['minq'])
    print("Entropy Regularization:", params['entropy_regularized'])
    
    print("{:=^100s}".format("Sampling Configurations"))
    print("Using quantile sampling during bootstrapping:", params['use_quantile_sampling_bootstrapping'])
    print("Using quantile sampling during evaluation interaction:", params['use_quantile_sampling_evaluation_interaction'])
    print("Using quantile sampling during training interaction:", params['use_quantile_sampling_training_interaction'])
    print("Sampling Percent:", params['quantile_sampling_percent'])
    print("Anneal Sampling Percent:", params['anneal_quantile_sampling'])
    
    print("{:=^100s}".format("Split"))
    """
    Define online and offline networks
    """
    Q_object = None
    Q_target = None
    
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
        train_env,
        state_size = train_env.observation_space.shape[0],
        action_size = train_env.action_space.shape[0],
        device=device,
        seed=args.seed
    )
    Q_target = Q_Constructor(
        params,
        train_env,
        state_size=train_env.observation_space.shape[0],
        action_size=train_env.action_space.shape[0],
        device=device,
        seed=args.seed
    )
    
    Q_target.eval()
    
    utils_for_q_learning.sync_networks(
        target=Q_target,
        online=Q_object,
        alpha=params['target_network_learning_rate'],
        copy=True)

    
    print("Start the Initialization Process!")
    steps = 0
    while steps < params["learning_starts"]:
        s, done, t = train_env.reset(), False, 0
        while not done:
            a = train_env.action_space.sample()
            s_, r, done, info = train_env.step(a)
            done_for_buffer = done and not info.get('TimeLimit.truncated',
                                                    False)
            Q_object.buffer_object.append(s, a, r, done_for_buffer, s_)
            
            s = s_
            steps += 1
    print("The initialization process finished!")
    
    steps = 0
    episodes = 0
    per_step_losses = []
    qs = []
    q_stars = []
    episodic_rewards = []
    iter_start_time = time()
    while steps < params["max_step"]:
        s, done = train_env.reset(), False
        
        episodic_reward = 0
        
        while not done:
            a = Q_object.enact_policy(s, episodes + 1, steps, 'train', params['policy_type'])
            s_, r, done, info = train_env.step(a)
            done_for_buffer = done and not info.get('TimeLimit.truncated', False)
            Q_object.buffer_object.append(s, a, r, done_for_buffer, s_)
            s = s_
            
            loss, basis_statistics = Q_object.update(Q_target, step=steps) 
            per_step_losses.append(loss)
            qs.append(basis_statistics["average_Q"])
            q_stars.append(basis_statistics["average_Q_star"])

            if ((steps % args.evaluation_frequency == 0) or (steps ==  params['max_step'] - 1)):
                evaluation_rewards = []
                for _ in range(10):
                    evaluation_reward = 0
                    s_eval, done_eval = eval_env.reset(), False
                    while not done_eval:
                        a_eval = Q_object.e_greedy_policy(s_eval, episodes + 1, steps, 'test')
                        s_eval_, r_eval, done_eval, _ = eval_env.step(a_eval)
                        evaluation_reward += r_eval
                        s_eval = s_eval_
                    evaluation_rewards.append(evaluation_reward)
                
                print(f"Step {steps}: Evaluation Reward: {np.mean(evaluation_rewards)}")
                
                """
                Consolidate episode statistics
                """
                mean_per_step_loss = np.nan_to_num(np.mean(np.array(per_step_losses)), nan=0)
                mean_qs = np.nan_to_num(np.mean(np.array(qs)), nan=0)
                mean_q_stars = np.nan_to_num(np.mean(np.array(q_stars)), nan=0)
                iter_total_time = time() - iter_start_time
                """
                Update meta logger to record some statistics
                """
                meta_logger.append_datapoint("evaluation_rewards", np.mean(evaluation_rewards), write=True)
                if episodic_rewards:
                    mean_episodic_reward = np.nan_to_num(np.mean(np.array(episodic_rewards)), nan=0)
                    meta_logger.append_datapoint("episodic_rewards", mean_episodic_reward, write=True)
                meta_logger.append_datapoint("episodes_so_far", episodes, write=True)
                meta_logger.append_datapoint("average_loss", mean_per_step_loss, write=True)
                meta_logger.append_datapoint("average_q", mean_qs, write=True)
                meta_logger.append_datapoint("average_q_star", mean_q_stars, write=True)
                meta_logger.append_datapoint("all_times", iter_total_time, write=True)

                """
                Reset tracking quantities
                """
                episodic_rewards, per_step_losses, qs, q_stars = [], [], [], []
                iter_start_time = time()


            if args.save_model and ((steps % 100000 == 0) or steps == (params['max_step'] - 1)):
                experiment_filepath = os.path.join(os.getcwd(), os.path.join(args.experiment_name, args.run_title))
                path = os.path.join(experiment_filepath, "logs")
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                torch.save(Q_object.state_dict(), os.path.join(path, f"seed_{args.seed}_object_" + str(steps)))
                torch.save(Q_target.state_dict(), os.path.join(path, f"seed_{args.seed}_target_" + str(steps)))
                    
            steps += 1
            episodic_reward += r
        
        episodes += 1
        episodic_rewards.append(episodic_reward)
        
