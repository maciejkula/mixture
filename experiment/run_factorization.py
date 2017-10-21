import argparse
import warnings

warnings.filterwarnings('error', module='torch.autograd')

import numpy as np

import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import mrr_score

from mixture.factorization_representation import MixtureNet


CUDA = torch.cuda.is_available()


def _evaluate(model, test, train):

    test_mrr = mrr_score(model, test, train=train)

    return test_mrr.mean()


def load_data(dataset, random_state):

    dataset = get_movielens_dataset(dataset)

    train, rest = random_train_test_split(dataset,
                                          random_state=random_state)

    test, validation = random_train_test_split(rest,
                                               test_percentage=0.5,
                                               random_state=random_state)

    return train, validation, test


if __name__ == '__main__':

    loss = 'adaptive_hinge'
    n_iter = 10
    learning_rate = 0.01
    l2 = 1e-9

    random_state = np.random.RandomState(42)

    train, validation, test = load_data('100K', random_state)

    representation = MixtureNet(train.num_users,
                                train.num_items,
                                num_components=4,
                                embedding_dim=32)
    model = ImplicitFactorizationModel(loss=loss,
                                       batch_size=512,
                                       representation=representation,
                                       learning_rate=learning_rate,
                                       n_iter=n_iter,
                                       l2=l2,
                                       use_cuda=CUDA,
                                       random_state=np.random.RandomState(42))
    model.fit(train, verbose=True)
    test_mrr = _evaluate(model, test, train.tocsr() + validation.tocsr())
    print('Test MRR {}'.format(test_mrr))
