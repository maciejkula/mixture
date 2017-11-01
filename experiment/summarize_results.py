from mixture.results import (generate_hyperparameter_table,
                             generate_performance_table,
                             plot_hyperparam_search,
                             plot_hyperparam_bootstrap,
                             read_results)


if __name__ == '__main__':

    results = read_results('mixture_results', 'sequence')

    plot_hyperparam_search(results)
    plot_hyperparam_bootstrap(results)

    print(generate_performance_table(results))
    print(generate_hyperparameter_table(results))
