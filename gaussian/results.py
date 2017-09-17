import numpy as np


def _get_type(trial):

    return trial['result']['hyper']['type']


def summarize_trials(trials):

    for model_type in ('pooling', 'lstm', 'gaussian', 'gaussian_kl'):
        results = [x for x in trials.trials if _get_type(x) == model_type]
        results = sorted(results, key=lambda x: x['result']['loss'])

        if results:
            print('Best {}: {}'.format(model_type, results[0]['result']))

        results = sorted(results, key=lambda x: -x['result']['test_mrr'])

        if results:
            print('Best test {}: {}'.format(model_type, results[0]['result']))
