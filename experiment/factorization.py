import argparse

import numpy as np

import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.goodbooks import get_goodbooks_dataset
from spotlight.datasets.amazon import get_amazon_dataset
from spotlight.cross_validation import random_train_test_split

from mixture.factorization import get_objective, hyperparameter_space
from mixture.hyperparameters import optimize
from mixture.results import summarize_trials


CUDA = torch.cuda.is_available()


def load_data(dataset, random_state):

    if 'goodbooks' in dataset:
        dataset = get_goodbooks_dataset()
    elif 'amazon' in dataset:
        dataset = get_amazon_dataset()
    else:
        dataset = get_movielens_dataset(dataset)

    train, rest = random_train_test_split(dataset,
                                          random_state=random_state)

    test, validation = random_train_test_split(rest,
                                               test_percentage=0.5,
                                               random_state=random_state)

    return train, validation, test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('dataset',
                        type=str,
                        help='Name of dataset to use')
    parser.add_argument('--num_trials',
                        default=10,
                        type=int,
                        help='Number of trials to run')

    args = parser.parse_args()

    random_state = np.random.RandomState(42)

    train, validation, test = load_data(args.dataset, random_state)

    print('Data {}'.format(train))

    objective = get_objective(train, validation, test)
    space = hyperparameter_space()

    fname = 'factorization_trials_{}.pickle'.format(args.dataset)

    for iteration in range(args.num_trials):
        print('Iteration {}'.format(iteration))
        trials = optimize(objective,
                          space,
                          trials_fname=fname,
                          max_evals=iteration + 1)

        summarize_trials(trials)
