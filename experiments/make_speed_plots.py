import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import json

COLOR_MAP = {
    # "DDPG": "dimgrey",
    # "Q-Functional Fourier Rank 1": "lightsalmon",
    # "Q-Functional Fourier Rank 2": "navy",
    # "Q-Functional Fourier Rank 3": "cornflowerblue",    
    "DDPG": "dimgrey",
    "Legendre Q-functional (Rank 1)": "lightsalmon",
    "Legendre Q-functional (Rank 2)": "navy",
    "Legendre Q-functional (Rank 3)": "cornflowerblue",
    "Legendre Q-functional (Rank 4)": "black",
    "Legendre Q-functional (Rank 3, Precompute)": "red",
}

keys = [
    "DDPG",
    "Legendre Q-functional (Rank 1)",
    "Legendre Q-functional (Rank 2)",
    "Legendre Q-functional (Rank 3)",
    "Legendre Q-functional (Rank 4)",
    "Legendre Q-functional (Rank 3, Precompute)"
]

def make_plots(filename, keys=None, save_path=None):
    plt.figure(figsize=(18, 11))
    with open(filename, "r") as f:
        data = json.load(f)

    if keys is None:
        keys = list(data.keys())

    # for key in ["DDPG", "Q-Functional Fourier Rank 1", "Q-Functional Fourier Rank 2", "Q-Functional Fourier Rank 3"]:
    for key in keys:
        speed_results = list(zip(*data[key]))
        # import ipdb; ipdb.set_trace()
        plt.plot(speed_results[0], speed_results[1], marker='o', label=key, linewidth=6, markersize=10, c=COLOR_MAP[key])

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Number of sampled actions", size=30)
    plt.ylabel("Seconds", size=30)
    plt.title("Time to sample actions", size=45)
    plt.legend(prop={"size": 24})

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # make_plots("experiments/speed_tests.json", save_path="experiments/long_speed_test.png")
    # make_plots("experiments/smaller_speed_tests.json", save_path="experiments/short_speed_test.png")

    make_plots("experiments/speed_test_results/raw/hopper.json", save_path="experiments/speed_test_results/plots/hopper_long.png")
    # make_plots("experiments/smaller_speed_tests.json", save_path="experiments/short_speed_test.png")
