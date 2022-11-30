import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("..")
sns.set()
from pathlib import Path

from general.plotting_utils import get_scores, generate_plot, get_all_run_titles, get_all_runs_and_subruns

from csv import DictReader

def get_dicts_from_filename(csv_filename):
    with open(csv_filename, 'r') as f:
        reader = DictReader(f)
        rows = [r for r in reader]

    return rows

def get_x_and_y_returns(csv_filename):
    rows = get_dicts_from_filename(csv_filename)
    rows = [r for r in rows if r['eval/mean_reward']]
    x = [float(r['time/total_timesteps']) for r in rows]
    y = [float(r['eval/mean_reward']) for r in rows]
    return x, y




def make_graphs(experiment_name,
                # subdir,
                run_titles=None,
                smoothen=False,
                min_length=-1,
                only_longest=False,
                skip_failures=False,
                cumulative=False,
                all_seeds=False,
                use_onager=False,
                final_min=None,
                final_max=None):

    if run_titles is None:
        print("Using all runs in experiment")
        run_titles = get_all_run_titles(experiment_name=experiment_name)
    run_titles = list(filter(lambda x: not "*" in x, run_titles))

    rts, scs = [], []
    for run_title in run_titles:
        scores = []
        for fname in glob.glob(os.path.join(experiment_name, run_title, "seed_*", "progress.csv"), recursive=True):
            x, y = get_x_and_y_returns(fname)
            scores.append(y)
        if not scores:
            continue

        shortest_score = min(len(s) for s in scores)
        scores = [s[:shortest_score] for s in scores]
        scores = np.array(scores)
        rts.append(run_title)
        scs.append(scores)

    [
        generate_plot(s, r, smoothen=smoothen)
        for s, r in zip(scs, rts)
    ]

    plt.legend()
    plt.title("First attempts at stable-baselines TD3")
    plt.show()


def main():
    """
    Change these options and directories to suit your needs
    """
    ## Defaults
    smoothen = True
    min_length = -1
    only_longest = False
    cumulative = False
    all_seeds = False

    experiment_name = "" # put the result folder here
    run_titles = get_all_run_titles(experiment_name)

    make_graphs(experiment_name,
                run_titles=run_titles,
                smoothen=smoothen)


if __name__ == '__main__':
    main()