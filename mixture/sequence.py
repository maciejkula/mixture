import time

from hyperopt import Trials, pyll, hp, fmin, STATUS_OK, STATUS_FAIL

import numpy as np

import torch

from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import PoolNet, LSTMNet

from mixture.model import DiversifiedImplicitSequenceModel
from mixture.representation import (MixtureLSTMNet,
                                    Mixture2LSTMNet,
                                    LinearMixtureLSTMNet,
                                    DiversifiedMixtureLSTMNet)


CUDA = torch.cuda.is_available()


def hyperparameter_space():

    common_space = {
        'batch_size': hp.quniform('batch_size', 64, 128, 16),
        'learning_rate': hp.loguniform('learning_rate', -6, -3),
        'l2': hp.loguniform('l2', -25, -9),
        'embedding_dim': 64,
        'num_components': hp.quniform('num_components', 2, 8, 2),
        'n_iter': hp.quniform('n_iter', 5, 50, 5),
        'loss': hp.choice('loss', ['adaptive_hinge'])
    }

    space = {
        'model': hp.choice('model', [
            # {
            #     'type': 'lstm',
            #     **common_space
            # },
            # {
            #     'type': 'mixture',
            #     **common_space
            # },
            # {
            #     'type': 'linear_mixture',
            #     **common_space
            # },
            {
                'type': 'diversified_mixture_fixed',
                'diversity_penalty': hp.uniform('diversity_penalty', 0, 2),
                **common_space
            },
        ])
    }

    return space


def get_objective(train_nonsequence, train, validation, test):

    def objective(hyper):

        print(hyper)

        start = time.clock()

        h = hyper['model']

        cls = ImplicitSequenceModel

        if h['type'] == 'pooling':
            representation = PoolNet(train.num_items,
                                     embedding_dim=int(h['embedding_dim']))
        elif h['type'] == 'lstm':
            representation = LSTMNet(train.num_items,
                                     embedding_dim=int(h['embedding_dim']))
        elif h['type'] == 'mixture':
            num_components = int(h['num_components'])
            embedding_dim = int(h['embedding_dim'])
            representation = MixtureLSTMNet(train.num_items,
                                            num_components=num_components,
                                            embedding_dim=embedding_dim)
        elif h['type'] == 'mixture2':
            num_components = int(h['num_components'])
            embedding_dim = int(h['embedding_dim'])
            representation = Mixture2LSTMNet(train.num_items,
                                             num_components=num_components,
                                             embedding_dim=embedding_dim)
        elif h['type'] == 'linear_mixture':
            num_components = int(h['num_components'])
            embedding_dim = int(h['embedding_dim'])
            representation = LinearMixtureLSTMNet(train.num_items,
                                                  num_components=num_components,
                                                  embedding_dim=embedding_dim)
        elif h['type'] == 'diversified_mixture_fixed':
            num_components = int(h['num_components'])
            embedding_dim = int(h['embedding_dim'])
            representation = DiversifiedMixtureLSTMNet(train.num_items,
                                                       num_components=num_components,
                                                       diversity_penalty=h['diversity_penalty'],
                                                       embedding_dim=embedding_dim)
            cls = DiversifiedImplicitSequenceModel
        else:
            raise ValueError('Unknown model type')

        model = cls(
            batch_size=int(h['batch_size']),
            loss=h['loss'],
            learning_rate=h['learning_rate'],
            l2=h['l2'],
            n_iter=int(h['n_iter']),
            representation=representation,
            use_cuda=CUDA,
            random_state=np.random.RandomState(42)
        )

        model.fit(train, verbose=True)

        elapsed = time.clock() - start

        print(model)

        validation_mrr = sequence_mrr_score(
            model,
            validation,
            exclude_preceding=True
        ).mean()
        test_mrr = sequence_mrr_score(
            model,
            test,
            exclude_preceding=True
        ).mean()

        print('MRR {} {}'.format(validation_mrr, test_mrr))

        if np.isnan(validation_mrr):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        return {'loss': -validation_mrr,
                'status': status,
                'validation_mrr': validation_mrr,
                'test_mrr': test_mrr,
                'elapsed': elapsed,
                'hyper': h}

    return objective
