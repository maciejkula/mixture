from mixture.results import (generate_hyperparameter_table,
                             generate_performance_table,
                             read_results)


if __name__ == '__main__':

    results = read_results('mixture_results', 'sequence')

    print(generate_performance_table(results))
    print(generate_hyperparameter_table(results))
