import matplotlib.pyplot as plt
import numpy as np


def plot_results():

    acc = np.zeros((8, 13))
    for node in range(8):
        path = f"./results/node {node}.npy"
        acc[node, :] = np.load(path)[:13]
        plt.plot(range(acc.shape[1]), acc[node, :], label=f"sat{node}")

    plt.legend()
    plt.xlabel("Training round")
    plt.ylabel("Test accuracy")
    plt.title("Walker (16 sats, 4 planes, 30 deg incl), 100 iterations/round")
    plt.grid()
    plt.show()


plot_results()
