import glob
import os
import pickle

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tabulate import tabulate


def _get_type(trial):

    return trial['result']['hyper']['type']


def summarize_trials(trials):

    for model_type in ('pooling', 'lstm',
                       'mixture', 'mixture2',
                       'linear_mixture', 'diversified_mixture',
                       'diversified_mixture_fixed',
                       'embedding_mixture',
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


    try:
        best = sorted(results, key=lambda x: x['result']['validation_mrr'])[-1]

        return best['result']['test_mrr']
    except IndexError:
        return '-'


def _get_test_result_history(results, model_type):

    results = [x for x in results.trials if _get_type(x) == model_type]

    return np.array([x['result']['test_mrr'] for x in results])


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
                                ('Mixture-LSTM', 'mixture'),
                                ('Linear Mixture-LSTM', 'linear_mixture')):

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


def plot_hyperparam_search(results, max_iter=100):

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    dataset_names = {'10M': 'Movielens 10M',
                     'amazon': 'Amazon',
                     'goodbooks': 'Goodbooks-10K'}

    for (dataset, ax) in zip(('10M', 'amazon', 'goodbooks'), (ax0, ax1, ax2)):

        baseline = np.maximum.accumulate(
            _get_test_result_history(results[dataset], 'lstm')[:max_iter])
        mixture = np.maximum.accumulate(
            _get_test_result_history(results[dataset], 'mixture')[:max_iter])

        ax.plot(np.arange(len(baseline)), baseline, label='LSTM')
        ax.plot(np.arange(len(mixture)), mixture, label='Mixture-LSTM')
        ax.set_title(dataset_names[dataset])

        if dataset == 'goodbooks':
            ax.legend()

    fig.tight_layout()
    fig.savefig('hyperparam_search.png')


def _bootstrap_max(data, num_samples):

    maxima = []

    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        maxima.append(np.max(sample))

    return np.array(maxima)


def plot_hyperparam_bootstrap(results, num_samples=10000):

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    dataset_names = {'10M': 'Movielens 10M',
                     'amazon': 'Amazon',
                     'goodbooks': 'Goodbooks-10K'}

    for (dataset, ax) in zip(('10M', 'amazon', 'goodbooks'), (ax0, ax1, ax2)):

        baseline = _get_test_result_history(results[dataset], 'lstm')
        mixture = _get_test_result_history(results[dataset], 'mixture')

        baseline_bootstrap = _bootstrap_max(baseline, num_samples)
        mixture_bootstrap = _bootstrap_max(mixture, num_samples)

        _, bins = np.histogram(np.concatenate([baseline_bootstrap,
                                               mixture_bootstrap]),
                               bins='auto')
        alpha = 0.75
        ax.hist(baseline_bootstrap,
                bins=bins,
                normed=True,
                alpha=alpha,
                label='LSTM')
        ax.hist(mixture_bootstrap,
                bins=bins,
                normed=True,
                alpha=alpha,
                label='Mixture-LSTM')
        ax.set_title(dataset_names[dataset])

        if dataset == 'goodbooks':
            ax.legend()

    fig.tight_layout()
    fig.savefig('hyperparam_bootstrap.png')
