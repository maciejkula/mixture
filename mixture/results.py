import glob
import os
import pickle

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tabulate import tabulate


sns.set_style('ticks', {'font.family': 'serif', 'axes.linewidth': 0.8})
sns.set_context('paper', font_scale=0.8)


class Result:

    def __init__(self, trials):

        self.trials = trials


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
        return 0.0


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


def generate_performance_table(sequence, factorization):

    headers = ['Model', 'Movielens 10M', 'Amazon', 'Goodbooks-10K']
    datasets = ('10M', 'amazon', 'goodbooks')
    rows = []

    for (model_name, model) in (('LSTM', 'lstm'),
                                ('Mixture-LSTM', 'mixture'),
                                ('Linear Mixture-LSTM', 'linear_mixture')):

        row = [model_name]

        for dataset in datasets:
            mrr = _get_best_test_result(sequence[dataset], model)
            row.append(mrr)

        rows.append(row)

    for (model_name, model) in (('Bilinear', 'bilinear'),
                                ('Projection Mixture', 'mixture'),
                                ('Embedding Mixture', 'embedding_mixture')):

        row = [model_name]

        for dataset in datasets:
            mrr = _get_best_test_result(factorization[dataset], model)
            row.append(mrr)

        rows.append(row)

    output = tabulate(rows,
                      headers=headers,
                      floatfmt='.4f',
                      tablefmt='latex_booktabs')

    return output.replace('Bilinear', '\midrule\n Bilinear')


def generate_hyperparameter_table(sequence, factorization):

    headers = ['Components', 'Movielens 10M', 'Amazon', 'Goodbooks-10K']

    outputs = []

    for (model, results) in (('mixture', sequence), ('embedding_mixture', factorization)):

        rows = []

        for num_components in (2, 4, 6, 8):

            row = [num_components]

            def filter_fnc(x):
                return x['result']['hyper'].get('num_components') == num_components

            for dataset in ('10M', 'amazon', 'goodbooks'):
                mrr = _get_best_test_result(results[dataset], model, filter_fnc)
                row.append(mrr)

            rows.append(row)

        output = tabulate(rows,
                          headers=headers,
                          floatfmt='.4f',
                          tablefmt='latex_booktabs')
        outputs.append(output)

    sequence_table, factorization_table = outputs

    return ('\\begin{table}\n'
            '\\caption{Effect of number of mixture components}\n'
            '\\label{tab:nummixtures}\n'
            '\\begin{subtable}{\\columnwidth}\n'
            '\\caption{Sequence models}\n'
            '%s\n'
            '\\end{subtable}\n'
            '\\begin{subtable}{\\columnwidth}\n'
            '\\caption{Factorization models}\n'
            '%s\n'
            '\\end{subtable}'
            '\\end{table}' % (sequence_table, factorization_table))


def plot_hyperparam_search(sequence, factorization, max_iter=100):

    fig, axes = plt.subplots(2, 3)

    sequence_axes, factorization_axes = (axes[0], axes[1])

    dataset_names = {'10M': 'Movielens 10M',
                     'amazon': 'Amazon',
                     'goodbooks': 'Goodbooks-10K'}

    for (i, (dataset, ax)) in enumerate(zip(('10M', 'amazon', 'goodbooks'),
                                            sequence_axes)):

        baseline = np.maximum.accumulate(
            _get_test_result_history(sequence[dataset], 'lstm')[:max_iter])
        mixture = np.maximum.accumulate(
            _get_test_result_history(sequence[dataset], 'mixture')[:max_iter])

        ax.plot(np.arange(len(baseline)), baseline, label='LSTM')
        ax.plot(np.arange(len(mixture)), mixture, label='Mixture-LSTM')
        ax.set_title(dataset_names[dataset])

        if i == 0:
            ax.set_xlabel('Iterations')
            ax.set_ylabel('MRR')

        if i == len(sequence_axes) - 1:
            ax.legend()

    for (i, (dataset, ax)) in enumerate(zip(('10M', 'amazon', 'goodbooks'),
                                            factorization_axes)):

        baseline = np.maximum.accumulate(
            _get_test_result_history(factorization[dataset], 'bilinear')[:max_iter])
        mixture = np.maximum.accumulate(
            _get_test_result_history(factorization[dataset], 'mixture')[:max_iter])
        embedding_mixture = np.maximum.accumulate(
            _get_test_result_history(factorization[dataset], 'embedding_mixture')[:max_iter])

        ax.plot(np.arange(len(baseline)), baseline, label='Bilinear')
        ax.plot(np.arange(len(mixture)), mixture, label='Projection Mixture')
        ax.plot(np.arange(len(embedding_mixture)), embedding_mixture, label='Embedding Mixture')
        ax.set_title(dataset_names[dataset])

        if i == 0:
            ax.set_xlabel('Iterations')
            ax.set_ylabel('MRR')

        if i == len(factorization_axes) - 1:
            ax.legend()

    fig.tight_layout()
    fig.savefig('hyperparam_search.eps')
