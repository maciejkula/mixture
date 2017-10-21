import glob
import os
import pickle

from tabulate import tabulate

import numpy as np


def _get_type(trial):

    return trial['result']['hyper']['type']


def summarize_trials(trials):

    for model_type in ('pooling', 'lstm',
                       'mixture', 'mixture2',
                       'linear_mixture', 'diversified_mixture',
                       'diversified_mixture_fixed',
                       'bilinear'):
        results = [x for x in trials.trials if _get_type(x) == model_type]
        results = sorted(results, key=lambda x: -x['result']['validation_mrr'])

        if results:
            print('Best {}: {}'.format(model_type, results[0]['result']))

        results = sorted(results, key=lambda x: -x['result']['test_mrr'])

        if results:
            print('Best test {}: {}'.format(model_type, results[0]['result']))


def _get_best_test_result(results, model_type, filter_fnc=None):

    if filter_fnc is None:
        filter_fnc = lambda x: True

    results = [x for x in results.trials
               if _get_type(x) == model_type and filter_fnc(x)]
    best = sorted(results, key=lambda x: x['result']['validation_mrr'])[-1]

    return best['result']['test_mrr']


def read_results(path, variant):

    filenames = glob.glob(os.path.join(path, '{}_trials_*.pickle'.format(variant)))

    results = {}

    for filename in filenames:
        with open(filename, 'rb') as fle:
            dataset_name = filename.split('_')[-1].replace('.pickle', '')
            data = pickle.load(fle)

            results[dataset_name] = data

    return results


def generate_performance_table(results):

    headers = ['Model', 'Movielens 10M', 'Amazon', 'Goodbooks-10K']
    rows = []

    for (model_name, model) in (('LSTM', 'lstm'),
                                ('Mixture-LSTM', 'mixture')):

        row = [model_name]

        for dataset in ('10M', 'amazon', 'goodbooks'):
            mrr = _get_best_test_result(results[dataset], model)
            row.append(mrr)

        rows.append(row)

    return tabulate(rows,
                    headers=headers,
                    floatfmt='.4f',
                    tablefmt='latex_booktabs')


def generate_hyperparameter_table(results):

    headers = ['Mixture components', 'Movielens 10M', 'Amazon', 'Goodbooks-10K']
    rows = []

    for num_components in (2, 4, 6, 8):

        row = [num_components]

        filter_fnc = lambda x: x['result']['hyper'].get('num_components') == num_components
        
        for dataset in ('10M', 'amazon', 'goodbooks'):
            mrr = _get_best_test_result(results[dataset], 'mixture', filter_fnc)
            row.append(mrr)

        rows.append(row)

    return tabulate(rows,
                    headers=headers,
                    floatfmt='.4f',
                    tablefmt='latex_booktabs')

