import argparse
import warnings

warnings.filterwarnings('error', module='torch.autograd')

import numpy as np

import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import LSTMNet
from spotlight.evaluation import sequence_mrr_score

from gaussian.model import DiversifiedImplicitSequenceModel
from gaussian.representation import (GaussianLSTMNet,
                                     GaussianKLLSTMNet,
                                     MixtureLSTMNet,
                                     Mixture2LSTMNet,
                                     LinearMixtureLSTMNet,
                                     DiversifiedMixtureLSTMNet)


CUDA = torch.cuda.is_available()


def _evaluate(model, test):

    test_mrr = sequence_mrr_score(model, test, exclude_preceding=True)

    return test_mrr.mean()


def load_data(dataset, random_state):

    dataset = get_movielens_dataset(dataset)

    # np.random.shuffle(dataset.timestamps)


    # max_sequence_length = int(np.percentile(dataset.tocsr()
    #                                         .getnnz(axis=1),
    #                                         80))
    max_sequence_length = 100
    min_sequence_length = 50
    step_size = max_sequence_length

    train_nonsequence, rest = user_based_train_test_split(dataset,
                                                          test_percentage=0.2,
                                                          random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)

    train = train_nonsequence.to_sequence(max_sequence_length=max_sequence_length,
                                          min_sequence_length=min_sequence_length,
                                          step_size=step_size)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                        min_sequence_length=min_sequence_length,
                                        step_size=step_size)

    return train_nonsequence, train, validation, test


if __name__ == '__main__':

    loss = 'bpr'
    n_iter = 1
    learning_rate = 0.01
    l2 = 1e-5

    random_state = np.random.RandomState(42)

    train_nonsequence, train, validation, test = load_data('100K', random_state)

    # model = ImplicitSequenceModel(loss=loss,
    #                               representation='pooling',
    #                               learning_rate=learning_rate,
    #                               l2=l2,
    #                               n_iter=n_iter,
    #                               use_cuda=CUDA,
    #                               random_state=np.random.RandomState(42))
    # model.fit(train, verbose=True)
    # test_mrr = _evaluate(model, test)
    # train_mrr = _evaluate(model, train)

    # print('Train {} test {}'.format(train_mrr, test_mrr))

    # model = ImplicitSequenceModel(loss=loss,
    #                               representation='lstm',
    #                               learning_rate=learning_rate,
    #                               l2=l2,
    #                               n_iter=n_iter,
    #                               use_cuda=CUDA,
    #                               random_state=np.random.RandomState(42))
    # model.fit(train, verbose=True)
    # test_mrr = _evaluate(model, test)
    # train_mrr = _evaluate(model, train)

    # print('Train {} test {}'.format(train_mrr, test_mrr))

    hyper = {'batch_size': 128.0,
             'embedding_dim': 62.0,
             'l2': 6.699791300878915e-05,
             'learning_rate': 0.014058539247016723,
             'loss': 'bpr',
             'n_iter': 5.0}

    hyper = {'batch_size': 80.0,
             'embedding_dim': 32.0,
             'l2': 0 * 5.645943698793739e-05,
             'learning_rate': 0.04822533874727729,
             'loss': 'adaptive_hinge',
             'n_iter': 1.0,
             'type': 'gaussian_kl'}

    # representation = GaussianKLLSTMNet(train.num_items,
                                       # embedding_dim=int(hyper['embedding_dim']))
    representation = DiversifiedMixtureLSTMNet(train.num_items,
                                               num_components=4,
                                               embedding_dim=int(hyper['embedding_dim']))
    model = DiversifiedImplicitSequenceModel(loss=hyper['loss'],
                                             batch_size=int(hyper['batch_size']),
                                             representation=representation,
                                             learning_rate=hyper['learning_rate'],
                                             n_iter=int(hyper['n_iter']),
                                             l2=hyper['l2'],
                                             use_cuda=CUDA,
                                             random_state=np.random.RandomState(42))
    model.fit(train, verbose=True)

    representation.train(False)
    test_mrr = _evaluate(model, test)
    print('Test MRR {}'.format(test_mrr))

