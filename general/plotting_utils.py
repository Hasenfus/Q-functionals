from pathlib import Path
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_scores(log_dir,
               subdir="scores",
               only_longest=False,
               min_length=-1,
               cumulative=False):
    longest = 0
    print(log_dir)
    overall_scores = []
    glob_pattern = log_dir + "/" + subdir + "/" + "*.pkl"
    for score_file in glob.glob(glob_pattern):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
        if only_longest:
            if len(scores) > longest:
                overall_scores = [scores]
                longest = len(scores)
        else:
            if len(scores) >= min_length:  # -1 always passes!
                overall_scores.append(scores)

    if len(overall_scores) == 0:
        raise Exception("No scores in " + log_dir)

    min_length = min(len(s) for s in overall_scores)

    overall_scores = [s[:min_length] for s in overall_scores]

    score_array = np.array(overall_scores)
    if cumulative:
        score_array = np.cumsum(score_array, axis=1)
    return score_array


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smoothen_data(scores, n=10):
    if len(scores) == 0 or len(scores[0]) == 0:
        return scores

    n = min(n, scores.shape[1])

    print(scores.shape)
    smoothened_cols = scores.shape[1] - n + 1
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data


def generate_plot(score_array, label, smoothen=False, return_line_objs=False, formatted_name=None):
    if smoothen:
        smooth_amount = 10 if isinstance(smoothen, bool) else smoothen
        score_array = smoothen_data(score_array, n=smooth_amount)
    median, mean, top, bottom = get_plot_params(score_array)

    if "Rank 1" in label:
        plt.plot(mean, linewidth=2, label=label, alpha=0.9, color="lightsalmon")
        plt.fill_between(range(len(top)), top, bottom, alpha=0.2, color="lightsalmon")
    elif "Rank 2" in label:
        plt.plot(mean, linewidth=2, label=label, alpha=0.9, color="navy")
        plt.fill_between(range(len(top)), top, bottom, alpha=0.2, color="navy")
    elif "Rank 3" in label:
        plt.plot(mean, linewidth=2, label=label, alpha=0.9, color="cornflowerblue")
        plt.fill_between(range(len(top)), top, bottom, alpha=0.2, color="cornflowerblue")
    elif "Rank 4" in label:
        plt.plot(mean, linewidth=2, label=label, alpha=0.9, color="slateblue")
        plt.fill_between(range(len(top)), top, bottom, alpha=0.2, color="slateblue")
    elif "Rank 5" in label:
        plt.plot(mean, linewidth=2, label=label, alpha=0.9, color="olive")
        plt.fill_between(range(len(top)), top, bottom, alpha=0.2, color="olive")
    else:
        p1 = None
        if formatted_name != None: 
            p1 = plt.plot(mean, linewidth=2, alpha=0.9, label=formatted_name)
        else: 
            p1 = plt.plot(mean, linewidth=2, label=label, alpha=0.9)
        p2 = plt.fill_between(range(len(top)), top, bottom, alpha=0.2)
        if (return_line_objs):
            return (p1, p2)

def get_all_run_titles(experiment_name):
    parent = Path(experiment_name)
    run_titles = [d.name for d in parent.iterdir()]
    print(run_titles)
    return run_titles


def get_all_runs_and_subruns(experiment_name):
    parent = Path(experiment_name)
    run_titles = []
    for d in parent.iterdir():
        if d.is_dir():
            run_titles.append(d.name)
            for sub_d in d.iterdir():
                if sub_d.is_dir():
                    run_titles.append(f"{d.name}/{sub_d.name}")
    return run_titles