import time

from hyperopt import Trials, pyll, hp, fmin, STATUS_OK, STATUS_FAIL

import numpy as np

import torch

from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet

from mixture.factorization_representation import MixtureNet


CUDA = torch.cuda.is_available()


def hyperparameter_space():

    common_space = {
        'batch_size': hp.quniform('batch_size', 128, 1024, 64),
        'learning_rate': hp.loguniform('learning_rate', -6, -3),
        'l2': hp.loguniform('l2', -25, -9),
        'embedding_dim': 64,
        'n_iter': hp.quniform('n_iter', 5, 50, 5),
        'loss': hp.choice('loss', ['adaptive_hinge'])
    }

    space = {
        'model': hp.choice('model', [
            {
                'type': 'bilinear',
                **common_space
            },
            {
                'type': 'mixture',
                'num_components': hp.quniform('num_components', 2, 8, 2),
                **common_space
            },
        ])
    }

    return space


def get_objective(train, validation, test):

    def objective(hyper):

        print(hyper)

        start = time.clock()

        h = hyper['model']

        cls = ImplicitFactorizationModel

        if h['type'] == 'bilinear':
            representation = BilinearNet(train.num_users,
                                         train.num_items,
                                         embedding_dim=int(h['embedding_dim']))
        elif h['type'] == 'mixture':
            representation = MixtureNet(train.num_users,
                                        train.num_items,
                                        num_components=int(h['num_components']),
                                        embedding_dim=int(h['embedding_dim']))
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

        try:
            model.fit(train, verbose=True)
        except ValueError:
            elapsed = time.clock() - start
            return {'loss': 0.0,
                    'status': STATUS_FAIL,
                    'validation_mrr': 0.0,
                    'test_mrr': 0.0,
                    'elapsed': elapsed,
                    'hyper': h}

        elapsed = time.clock() - start

        print(model)

        validation_mrr = mrr_score(
            model,
            validation,
            train=(train.tocsr() + test.tocsr())
        ).mean()
        test_mrr = mrr_score(
            model,
            test,
            train=(train.tocsr() + validation.tocsr())
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
