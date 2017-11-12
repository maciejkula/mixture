import os

from mixture.results import (generate_dataset_table,
                             generate_hyperparameter_table,
                             generate_performance_table,
                             plot_hyperparam_search,
                             read_results)


if __name__ == '__main__':

    dest_dir = os.path.abspath('paper/includes')

    sequence = read_results('mixture_results', 'sequence')
    factorization = read_results('mixture_results', 'factorization')

    fig = plot_hyperparam_search(sequence, factorization, max_iter=20)
    fig.savefig(os.path.join(dest_dir, 'hyperparam_search.pdf'))

    with open(os.path.join(dest_dir, 'performance.tex'), 'w') as fle:
        fle.write(generate_performance_table(sequence, factorization))

    with open(os.path.join(dest_dir, 'hyperparameters.tex'), 'w') as fle:
        fle.write(generate_hyperparameter_table(sequence, factorization))

    with open(os.path.join(dest_dir, 'datasets.tex'), 'w') as fle:
        fle.write(generate_dataset_table())
