import collections
import logging
import pprint

import torch

from algs.activeLearning import ucb
from algs.gp import GPModel
from arguments import get_args

import numpy as np
from topo_data_util.embedding import embed_data

from util import feed_random_seeds


def compute_mse(predictor, test_x, test_y):
    data_size = len(test_x)
    if data_size == 0:
        return None

    return np.mean([(predictor(test_x[idx]) - test_y[idx].numpy()) ** 2 for idx in range(data_size)])

def compute_rse(predictor, test_x, test_y):
    """
    :return: sum (predictor(x) - test_y(x))^2 / sum (test_y(x) - mean_test_y)^2
    """
    if len(test_y) == 0: return 0

    data_size = len(test_x)

    predicts = [predictor(test_x[idx]) for idx in range(data_size)]
    pprint.pprint(list(zip(predicts, test_y.numpy())))

    numerator = sum((predicts[idx] - test_y[idx].numpy()) ** 2 for idx in range(data_size))

    test_y_mean = np.mean(test_y.numpy())
    denominator = sum((test_y[idx].numpy() - test_y_mean) ** 2 for idx in range(data_size))

    return 1. * numerator / denominator

def compute_mse_by_target(predictor, test_x, test_y, bin_size=10, maximum=50):
    """
    Divide test_y into bins, report their rse separately
    """
    range_from = 0
    range_to = bin_size
    results = []

    while range_from < maximum:
        indices = torch.nonzero(torch.logical_and(range_from <= test_y, test_y <= range_to + 0.001))
        indices = indices.flatten()

        results.append(compute_mse(predictor, test_x[indices], test_y[indices]))

        range_from += bin_size
        range_to += bin_size

    return results


def train_gp(args, input, target, output):
    feed_random_seeds(args.seed)

    train_size = 3000
    valid_size = 0
    test_size = 700

    total_size = train_size + valid_size + test_size

    data_x, data_y, vec_of_paths = embed_data(filename=input, size_needed=total_size, key=target, embed='freq')
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    print('data size', len(data_x))

    # slicing index
    train_end = train_size
    valid_end = train_size + valid_size
    test_end = train_size + valid_size + test_size

    exp_times = 1

    for exp_time in range(exp_times):
        # shuffle data
        shuffled = np.random.permutation(total_size)
        data_x = data_x[shuffled]
        data_y = data_y[shuffled]

        # slice data set
        train_x = torch.Tensor(data_x[:train_end]) # |train_size| * |bag of paths|
        train_y = torch.Tensor(data_y[:train_end]) # |train_size|
        valid_x = torch.Tensor(data_x[train_end:valid_end])
        valid_y = torch.Tensor(data_y[train_end:valid_end])
        test_x = torch.Tensor(data_x[valid_end:test_end])
        test_y = torch.Tensor(data_y[valid_end:test_end])

        # fit a gp
        gp = GPModel(train_x, train_y, sigma=args.sigma)
        with torch.no_grad():
            gp_predictor = lambda x: gp.get_mean(x)
            print('rse', compute_rse(gp_predictor, test_x, test_y))

            print(compute_mse_by_target(gp_predictor, test_x, test_y))

            baseline = lambda x: torch.mean(test_y)
            print('baseline mse', compute_mse(baseline, test_x, test_y))

            # save the (last) learned gp model
            torch.save({'model_state_dict': gp.model.state_dict(),
                        'train_x': train_x,
                        'train_y': train_y,
                        'vec_of_paths': vec_of_paths},
                       output + '.pt')


if __name__ == '__main__':
    args = get_args()

    train_gp(args, input='4comp_old.json', target='eff', output='efficiency')
    train_gp(args, input='4comp_old.json', target='vout', output='vout')
