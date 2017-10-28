import argparse
import warnings

warnings.filterwarnings('error', module='torch.autograd')

import numpy as np

import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
# from spotlight.factorization.representations import BilinearNet
from spotlight.evaluation import mrr_score

from mixture.factorization_representation import MixtureNet, NonlinearMixtureNet, BilinearNet, EmbeddingMixtureNet


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

    hyper = {'batch_size': 192.0 * 10,
             'embedding_dim': 64,
             'l2': 2.01514921329504e-11,
             'learning_rate': 0.004503611222494057 * 1e-2,
             'loss': 'adaptive_hinge',
             'n_iter': 10.0,
             'num_components': 6.0,
             'type': 'mixture'}

    hyper = {'batch_size': 192,
             'embedding_dim': 64,
             'l2': 2.01514921329504e-11,
             'learning_rate': 0.004503611222494057 * 1e0,
             'loss': 'adaptive_hinge',
             'n_iter': 10.0,
             'num_components': 4.0,
             'type': 'mixture'}
    hyper = {'batch_size': 16384.0,
             'embedding_dim': 32,
             'l2': 2.0074425248828707e-07,
             'learning_rate': 0.0048015875347904155,
             'loss': 'adaptive_hinge',
             'n_iter': 100.0,
             'num_components': 3.0,
             'type': 'mixture'}

    train, validation, test = load_data('100K', random_state)

    representation = EmbeddingMixtureNet(train.num_users,
                                         train.num_items,
                                         num_components=int(hyper['num_components']),
                                         embedding_dim=int(hyper['embedding_dim']))
    # representation = BilinearNet(train.num_users,
    #                              train.num_items,
    #                              embedding_dim=int(hyper['embedding_dim']))
    model = ImplicitFactorizationModel(loss=hyper['loss'],
                                       batch_size=int(hyper['batch_size']),
                                       representation=representation,
                                       learning_rate=hyper['learning_rate'],
                                       n_iter=int(hyper['n_iter']),
                                       l2=hyper['l2'],
                                       use_cuda=CUDA,
                                       random_state=np.random.RandomState(42))
    model.fit(train, verbose=True)
    model._net.train(False)
    test_mrr = _evaluate(model, test, train.tocsr() + validation.tocsr())
    print('Test MRR {}'.format(test_mrr))
    print(model)
