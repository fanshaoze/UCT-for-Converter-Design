import math


def ucb(means, variances, query_time):
    """
    Logic of ucb selection.
    Return the id of bandit selected given bandits and the current select time (starting from 1)
    """
    size = len(means)
    # ucb exploration weight
    beta = math.sqrt(2 * math.log(size * query_time ** 2 * math.pi ** 2 / (6 * 0.05)))

    bandits = [mean + beta * math.sqrt(var) for mean, var in zip(means, variances)]
    return max(range(size), key=lambda x: bandits[x])
