import os
import sys
from collections import defaultdict, OrderedDict
import gym
import numpy as np

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("String '{}' is not a known bool value.".format(s))


def autoconvert(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


def update_param(params, arg_name, arg_value):
    if arg_name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".
            format(arg_name))
    else:
        print("Updating parameter '{}' to {}".format(arg_name, arg_value))
    converted_arg_value = autoconvert(arg_value)
    if type(params[arg_name]) != type(converted_arg_value):
        error_str = f"Old and new type must match! Got {type(converted_arg_value)}, expected {type(params[arg_name])}, for {arg_name}"
        raise ValueError(error_str)
    params[arg_name] = converted_arg_value


def get_hyper_parameters(name, hyper_param_directory):
    meta_params = {}
    filepath = os.path.join(hyper_param_directory, f"{name}.hyper")

    with open(filepath) as f:
        lines = [line.rstrip('\n') for line in f]
        for l in lines:
            parameter_name, parameter_value, parameter_type = (l.split(','))
            if parameter_type == 'string':
                meta_params[parameter_name] = str(parameter_value)
            elif parameter_type == 'integer':
                meta_params[parameter_name] = int(parameter_value)
            elif parameter_type == 'float':
                meta_params[parameter_name] = float(parameter_value)
            elif parameter_type == 'boolean':
                meta_params[parameter_name] = boolify(parameter_value)
            else:
                print("unknown parameter type ... aborting")
                print(l)
                sys.exit(1)
    return meta_params


def save_hyper_parameters(params, seed):
    assert "hyperparams_dir" in params
    assert "hyper_parameters_name" in params

    hyperparams_filename = '{}__seed_{}.hyper'.format(
        params['hyper_parameters_name'],
        seed,
    )
    hyperparams_path = os.path.join(params['hyperparams_dir'],
                                    hyperparams_filename)
    with open(hyperparams_path, 'w') as file:
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                int: 'integer',
                str: 'string',
                float: 'float',
                bool: 'boolean',
            })[type(value)] # yapf: disable
            if type_str is not None:
                file.write("{},{},{}\n".format(name, value, type_str))


def cuda_or_cpu(s):
    if s not in ("cuda", "cpu"):
        raise ValueError(f"must pass 'cuda' or 'cpu'. Got {s}")
    return str(s)



class DMSuiteUnwrapper(gym.Wrapper):
    """
    Makes observation space correct as well, so the whole interface is correct
    """
    def __init__(self, env):
        super(DMSuiteUnwrapper, self).__init__(env)
        self.observation_space = self.observation_space['observations']
        # required for compatibility with stable-baselines
        self.action_space.dtype = np.dtype('float32')
        # Because otherwise we don't get the proper done signal!
        self._max_episode_steps = 999

    def reset(self, *args, **kwargs):
        state = super(DMSuiteUnwrapper, self).reset(*args, **kwargs)
        assert isinstance(state, OrderedDict)
        return state['observations']

    def step(self, *args, **kwargs):
        state, reward, done, info = super(DMSuiteUnwrapper,
                                          self).step(*args, **kwargs)
        assert isinstance(state, OrderedDict)
        return state['observations'], reward, done, info

    def render(self, *args, **kwargs):
        kwargs['use_opencv_renderer'] = True
        return super(DMSuiteUnwrapper, self).render(*args, **kwargs)


class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=0.1):
        super(ScaleRewardWrapper, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


def make_env(env_name, *args, **kwargs):
    return gym.make(env_name)
