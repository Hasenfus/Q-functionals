from cmath import isnan
import dis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gym
import sys
sys.path.append("..")
import copy
import torch
from gym import spaces
from itertools import combinations
from scipy.stats import multivariate_normal, norm
from scipy.special import softmax
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from functional_critic.agents import FourierAgent, LegendreAgent, PolynomialAgent
from functional_critic import utils, utils_for_q_learning
import argparse
from time import sleep
import os
import seaborn as sns
sns.set()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MultimodalEnv(gym.Env):
    def __init__(self, s_dim = 1, a_dim = 1, std = 1, scale=1, center=0.5):
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(s_dim, ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(a_dim, ))
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.std = std
        self.scale = scale
        self.center = center
        
        self.distributionCenters = []
        for count in range(0, self.a_dim + 1):
            Combs = combinations(range(0, self.a_dim), count)
            for indices in Combs:
                center = np.ones((self.a_dim)) * self.center
                for index in indices:
                    center[index] = -self.center
                self.distributionCenters.append(center)
        # breakpoint()
        self.distributionCenters = np.array(self.distributionCenters)
        self.distributions = []
        # breakpoint()
        for center in self.distributionCenters:
            var = multivariate_normal(mean=center, cov=self.std)
            self.distributions.append(var)
        
    def reset(self):
        # Since it's a bandit, I think we should have a constant state
        return np.ones((self.s_dim,)).astype(np.float32)
    
    def step(self, action):
        # assert len(action) == self.a_dim
        reward = 0
        for distribution in self.distributions:
            reward += distribution.pdf(action) * self.scale
        reward = int(reward * 1000) / 1000 # only keep 3 digits after the decimal point
        done = True
        # print(reward)
        return np.zeros((self.s_dim,)).astype(np.float32), reward, done, {}
    
def plot_reward():
    train_env = MultimodalEnv(s_dim=5, a_dim=1, std=0.1, scale=1, center=1)
    actions = np.arange(-2, 2, 0.0025)
    rewards = [train_env.step(action)[1] for action in actions]
    plt.plot(actions, rewards)
    plt.show()

def plot_qfunctional():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="../hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameters_name",
                        required=False,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="00")  # OpenAI gym environment name
    parser.add_argument("--env_std",
                        required=True,
                        default=0.025,
                        type=float)
    parser.add_argument("--env_mean",
                        required=True,
                        default=0.5,
                        type=float)
    
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
    
    
    """
    The parameters controlling the environment
    """
    params['seed'] = args.seed

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    train_env = MultimodalEnv(s_dim=5, a_dim=1, std=args.env_std, scale=1, center=args.env_mean)
    eval_env = MultimodalEnv(s_dim=5, a_dim=1, std=args.env_std, scale=1, center=args.env_mean)
    
    params['env'] = train_env
    utils_for_q_learning.set_random_seed(params)
    
    command_string = '"python ' + " ".join(sys.argv) + '"'
    params["command_string"] = command_string

    
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
    params["batch_size"] = 32
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
    while steps < 50:
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
    while steps < 1000:
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

            if ((steps % 50 == 0) or (steps ==  params['max_step'] - 1)):
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
    # breakpoint()
    actions = np.arange(-args.env_mean*2, args.env_mean*2, 0.0025).reshape(-1, 1).astype(np.float32)
    state = np.ones((actions.shape[0], 5), dtype=np.float32())
    action_values = Q_object.forward(torch.tensor(state), torch.tensor(actions)).detach().numpy()
    return action_values, args.env_mean, args.env_std, params["target_network_learning_rate"]

def plot_policy():
    
    action_values, env_mean, env_std, tau= plot_qfunctional()
    actions = np.arange(-env_mean*2, env_mean*2, 0.0025)
    # breakpoint()
    probs = np.exp(action_values)/np.exp(action_values).sum()
    plt.plot(actions, probs, label="Q-Functional (Fourier, Rank 3)", linewidth=1.9, color="royalblue")
   
    train_env = MultimodalEnv(s_dim=5, a_dim=1, std=env_std, scale=1, center=env_mean)
    test_env = MultimodalEnv(s_dim=5, a_dim=1, std=env_std, scale=1, center=env_mean)
    noise = NormalActionNoise(mean=0, sigma=0.01)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[64, 64], qf=[64, 64]))
    model = SAC("MlpPolicy", train_env, learning_rate=0.01, buffer_size=50, learning_starts=50,
                batch_size=32, tau=tau, action_noise=noise, replay_buffer_class=ReplayBuffer, \
                ent_coef=1, verbose=0, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=1050, eval_env=test_env, eval_freq=100, n_eval_episodes=5)

    # code for plotting SAC policy
    ob = torch.tensor(train_env.reset())
    feature_extractor = model.policy.actor.latent_pi
    mu = model.policy.actor.mu
    log_std = model.policy.actor.log_std
    distribution = model.policy.actor.action_dist
    
    features = feature_extractor(ob)
    mean = mu(features)
    log_std = torch.clamp(log_std(features), LOG_STD_MIN, LOG_STD_MAX)
    distribution = distribution.proba_distribution(mean, log_std)

    actions = np.arange(-env_mean*2, env_mean*2, 0.0025)
    values = []
    for i in range(actions.shape[0]):
        log_prob_prob_action = distribution.log_prob(torch.tensor([[[actions[i]]]]))
        action_prob = 0 if np.isnan(torch.exp(log_prob_prob_action).detach().numpy()[0, 0]) else torch.exp(log_prob_prob_action).detach().numpy()[0, 0]
        values.append(action_prob)
    values = np.array(values)/sum(values)
    
    plt.plot(actions, values, label="SAC", linewidth = 1.9, color="salmon")
    print("Sum of values: ", sum(values))

    # code for plotting optimal policy
    env = MultimodalEnv(s_dim=5, a_dim=1, std=env_std, scale=env_mean)
    actions = np.arange(-env_mean*2, env_mean*2, 0.0025)
    values = []
    for i in range(actions.shape[0]):
        values.append(env.step(np.array([actions[i]]))[1])
    values = np.array(values)
    policy = softmax(values, axis=0)
    # policy = values/10
    plt.plot(actions, policy, label="Optimal", linewidth=1.9, color="gold")
    print("Sum of policy: ", sum(policy))
    # plt.show()
    plt.title("Derived Policies of Q-functionals and SAC")
    plt.xlabel("Action")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

def plot_learning():
    # The best possible possible value
    env = MultimodalEnv(s_dim=5, a_dim=1, std=0.025, scale=1)
    actions = np.arange(-1, 1, 0.0025)
    values = []
    for i in range(actions.shape[0]):
        values.append(env.step(np.array([actions[i]]))[1])
    values = np.array(values)
    # print('divided by 2!')
    policy = softmax(values, axis=0)# / 2.

    
    summation = 0
    for idx, item in enumerate(policy):
        summation += item*env.step(np.array([actions[idx]]))[1] # calculation for the expected reward
    for idx, item in enumerate(policy):
        summation += item *(-np.log(item))

    best_summation = summation
    # The learning process of SAC
    train_env = MultimodalEnv(s_dim=5, a_dim=1, std=0.025, scale=1)
    test_env = MultimodalEnv(s_dim=5, a_dim=1, std=0.025, scale=1)
    noise = NormalActionNoise(mean=0, sigma=1.0)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[1], qf=[64, 64]))
    
    model = SAC("MlpPolicy", train_env, learning_rate=0.001, buffer_size=50, learning_starts=0,
                batch_size=50, tau=0.005, action_noise=noise, replay_buffer_class=ReplayBuffer, \
                ent_coef=0.01, verbose=1, policy_kwargs=policy_kwargs)
    
    sums = []
    for _ in range(5000//50):
        model.learn(total_timesteps=50, eval_env=test_env, eval_freq=-1, n_eval_episodes=5)
        ob = torch.tensor(env.reset())
        with torch.no_grad():
            feature_extractor = copy.deepcopy(model.policy.actor.latent_pi)
            mu = copy.deepcopy(model.policy.actor.mu)
            log_std = copy.deepcopy(model.policy.actor.log_std)
            distribution = copy.deepcopy(model.policy.actor.action_dist)
            features = feature_extractor(ob)
            mean = mu(features)
            log_std = torch.clamp(log_std(features), LOG_STD_MIN, LOG_STD_MAX)
    
            distribution = distribution.proba_distribution(mean, log_std)
        actions = np.arange(-1, 1, 0.0025)
        policy = []
        for i in range(actions.shape[0]):
            log_prob_prob_action = distribution.log_prob(torch.tensor([[[actions[i]]]]))
            policy.append(torch.exp(log_prob_prob_action).detach().numpy()[0, 0])
        policy = np.array(policy)/sum(policy)
        summation = 0
        for idx, item in enumerate(policy):
            summation += item*env.step(np.array([actions[idx]]))[1] # calculation for the expected reward
        for idx, item in enumerate(policy):
            summation += item *(-np.log(item))
        sums.append(summation)

    here_actions = np.arange(-1, 1, 0.0025)
    rewards = [train_env.step(action)[1] for action in actions]
    print('max policy: ', max(policy))

    plt.plot(actions, rewards, label="rewards")
    plt.plot(actions, policy*5000); plt.show()

    plt.plot(sums)
    plt.axhline(y=best_summation, color='r', linestyle='-')
    plt.show()


    
plot_policy()
# plot_learning()
# plot_reward()