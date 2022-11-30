import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gym
from gym import spaces
import torch
from itertools import combinations
from scipy.stats import multivariate_normal

def z_func(x, y, cos_mag_scale=3., cos_freq_scale=5., norm_scale=1.):
    return cos_mag_scale * np.cos(cos_freq_scale * x) * np.cos(
        cos_freq_scale * y) - norm_scale * (np.abs(x) + np.abs(y))



class SineRewardEnv(gym.Env):
    """
    This guy is a bit different. What's different? |a|_2 (\sum_i sin(a_i)). But maybe more interestingly,
    there will be state AND action within those things. So, the state is given to you, and the
    action you need to choose yourself. So it's not a bandit anymore. That's actually a much better test-bed. We can do
    standard RBFDQN, latent RBFDQN, and DDPG!
    """
    def get_reward(self, s, a):
        assert s.shape == (self.s_dim, ), f"{s.shape} {self.s_dim}"
        assert a.shape == (self.a_dim, ), f"{a.shape} {self.a_dim}"
        magnitude = np.linalg.norm(np.concatenate([s, a]))
        assert magnitude.shape == (), "Expects a single number"
        sines_1 = np.sin(s*self.FREQ_SCALE)
        sines_2 = np.sin(a*self.FREQ_SCALE)
        sines_sum = (np.sum(sines_1) + np.sum(sines_2)) / (self.s_dim + self.a_dim)
        return magnitude * sines_sum

    def __init__(self, s_dim=1, a_dim=1, episode_length=float('inf')):
        """
        Action is what makes reward
        Observation is always the same.
        For now, episode_length is determined through the registration thing in init
        We'll have a special case where s_dim being zero makes the observation space constant and zero.
        """
        was_zero = False
        if s_dim <= 0:
            s_dim = 1
            was_zero = True

        self.s_dim = s_dim
        self.a_dim = a_dim
        print(f"sdim: {s_dim}\t adim: {a_dim}")
        self.episode_length = episode_length
        self.FREQ_SCALE = 4.

        if was_zero:
            self.observation_space = spaces.Box(0.0, 0.0, shape=(self.s_dim, ))
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.s_dim, ))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.a_dim, ))


        self._last_state = None
        self._t = 0

        self.COS_MAG_SCALE = 3.
        self.COS_FREQ_SCALE = 5.
        self.NORM_SCALE = 1.

    def reset(self):
        self._last_state = self.observation_space.sample()
        self._t = 0
        return self._last_state.copy()

    def step(self, action):
        reward = self.get_reward(self._last_state, action)
        assert self.action_space.contains(action), action
        self._last_state = self.observation_space.sample()
        self._t += 1
        done = (self._t >= self.episode_length)

        return self._last_state.copy(), reward, done, {}

    def plot(self, savefile):
        assert self.a_dim == 2
        x = np.arange(-1.0, 1.0, 0.05)
        y = np.arange(-1.0, 1.0, 0.05)
        X, Y = np.meshgrid(x, y)  # grid of point
        actions = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        states = np.zeros((actions.shape[0], self.s_dim))
        Z = np.array([self.get_reward(states[i], actions[i]) for i in range(len(actions))])
        Z = Z.reshape(X.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X,
                               Y,
                               Z,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(savefile)
        plt.close()

    def plot_agent(self, Q_object, savefile):
        assert self.a_dim == 2
        x = np.arange(-1.0, 1.0, 0.05)
        y = np.arange(-1.0, 1.0, 0.05)
        X, Y = np.meshgrid(x, y)  # grid of point
        actions = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        actions_torch = torch.Tensor(actions).to(Q_object.device)

        states_torch = torch.zeros(actions.shape[0], self.s_dim).to(Q_object.device)
        Z_torch = Q_object.forward(states_torch, actions_torch)
        Z = Z_torch.detach().cpu().numpy()
        Z = Z.reshape(X.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X,
                               Y,
                               Z,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(savefile)
        plt.close()
        
        
class MultimodalEnv(gym.Env):
    def __init__(self, s_dim = 1, a_dim = 1, std = 1, scale=1):
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(s_dim, ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(a_dim, ))
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.std = std
        self.scale = scale
        
        self.distributionCenters = []
        for count in range(0, self.a_dim + 1):
            Combs = combinations(range(0, self.a_dim), count)
            for indices in Combs:
                center = np.ones((self.a_dim)) * 0.5
                for index in indices:
                    center[index] = -0.5
                self.distributionCenters.append(center)
        self.distributionCenters = np.array(self.distributionCenters)
        self.distributions = []
        for center in self.distributionCenters:
            var = multivariate_normal(mean=center, cov=self.std)
            self.distributions.append(var)
        
    def reset(self):
        # Since it's a bandit, we should have a constant state
        return np.ones((self.s_dim,))
    
    def step(self, action):
        reward = 0
        for distribution in self.distributions:
            reward += distribution.pdf(action) * self.scale
        reward = int(reward * 1000) / 1000 # only keep 3 digits after the decimal point
        done = True
        return np.zeros((self.s_dim,)), reward, done, {}

if __name__ == "__main__":
    env = MultimodalEnv(s_dim=1, a_dim=2, std=0.1, scale = 0.5)
    breakpoint()