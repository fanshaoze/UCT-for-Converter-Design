import sys

import gpytorch
import torch

from matplotlib import pyplot as plt
import numpy as np

#PLOT_BASE = 'figures/'
PLOT_BASE = ''


def plot_errorbars(x_ticks, x_values, y_values, y_errors, x_label, y_label, title, filename):
    if x_values is None:
        x_values = range(y_values)

    plt.figure(figsize=(4, 3))

    plt.bar(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim([min(x_values) - 1, max(x_values) + 1])
    plt.ylim([0, 2])

    if x_ticks is not None:
        plt.xticks(x_values, x_ticks)

    plt.tight_layout()

    plt.savefig(PLOT_BASE + filename + ".png", dpi=300, format="png")

    plt.close()

def plot_hist(values, x_label, title, filename):
    plt.figure(figsize=(4, 3))

    plt.hist(values.numpy(), bins=np.arange(-20, 21) * .05)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(filename + ".pdf", dpi=300, format="pdf")

    plt.close()

def sort_data_by_gap(data1, data2):
    diff = (data1 - data2).numpy()
    sorted_indices = np.argsort(diff)

    print("best cases")
    print('[' + ','.join(map(lambda _: str(_), sorted_indices[-10:])) + ']')
    print("worst cases")
    print('[' + ','.join(map(lambda _: str(_), sorted_indices[:10])) + ']')

def plot_results(filename):
    data = torch.load(filename)
    algorithm_names = ["EPU", "UCB", "Mean", "Uncertain"]

    evals = {"posterior": lambda x: x["posterior_values"],
             #"time": lambda x: x["times"],
             "evoi": lambda x: x["evois"]}

    eval_labels = {"posterior": "Posterior Value",
                   "time": "Computation Time (sec.)",
                   "evoi": "Value Improvement"}

    for eval_name, eval_func in evals.items():
        stat_data = eval_func(data)

        print(eval_name)
        #print(stat_data.numpy())
        mean = torch.mean(stat_data, dim=0).numpy()
        std = torch.std(stat_data, dim=0).numpy()
        print(mean)
        print(std)

        batch = 5
        for alg_idx, alg_name in enumerate(algorithm_names):
            left = alg_idx * batch
            right = (alg_idx + 1) * batch
            plot_errorbars(None, range(left+1, right+1), mean[left:right], std[left:batch],
                           "Number of Queries", eval_labels[eval_name], "", eval_name + "_" + alg_name)

    #for column_id in range(len(algorithm_names)):
    #    plot_hist(post_values[:,column_id], x_label="Posterior - Prior", title=algorithm_names[column_id], filename="hist_" + algorithm_names[column_id])

def plot_data_points(x_values, y_values, x_label, y_label, filename):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    if torch.is_tensor(x_values): x_values = x_values.numpy()
    if torch.is_tensor(y_values): y_values = y_values.numpy()

    plt.plot(x_values, y_values)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()

    plt.savefig(PLOT_BASE + filename + ".png", dpi=300, format="png")

    plt.close()

def bar_plot(x_values, y_values, x_label, y_label, filename):
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        plt.bar(x_values, y_values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.savefig(filename + ".pdf", dpi=300, format="pdf")

        plt.close()

def plot_heatmap(grid, filename, extent=None, points=None, path=None, query_data_size=0):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    plt.imshow(grid, extent=extent)
    plt.colorbar()

    if points is not None:
        data_size = points.size()[0]
        # training data
        plt.plot(points[:data_size - query_data_size, 0], points[:data_size - query_data_size, 1], 'k*')
        # queries
        plt.plot(points[data_size - query_data_size:, 0], points[data_size - query_data_size:, 1], 'r*')

    if path is not None:
        plt.plot(path[:, 0], path[:, 1], 'b')

    plt.savefig(PLOT_BASE + filename + ".png", dpi=300, format="png")

    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("need to specify pt file to be plotted.")
    else:
        plot_results(sys.argv[1])
