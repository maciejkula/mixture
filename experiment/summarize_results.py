from mixture.results import (generate_hyperparameter_table,
                             generate_performance_table,
                             plot_hyperparam_search,
                             read_results)


if __name__ == '__main__':

    sequence = read_results('mixture_results', 'sequence')
    factorization = read_results('mixture_results', 'factorization')

    plot_hyperparam_search(sequence, factorization, max_iter=25)

    print(generate_performance_table(sequence, factorization))
    # print(generate_hyperparameter_table(results))
