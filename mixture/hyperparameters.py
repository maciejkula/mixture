import os

import pickle

import shutil

import hyperopt
from hyperopt import Trials, fmin


def optimize(objective, space, trials_fname=None, max_evals=5):

    if trials_fname is not None:
        if os.path.exists(trials_fname):
            with open(trials_fname, 'rb') as trials_file:
                trials = pickle.load(trials_file)
        else:
            trials = Trials()

    fmin(objective,
         space=space,
         algo=hyperopt.tpe.suggest,
         #algo=hyperopt.rand.suggest,
         trials=trials,
         max_evals=max_evals)

    if trials_fname is not None:
        temporary = '{}.temp'.format(trials_fname)
        with open(temporary, 'wb') as trials_file:
            pickle.dump(trials, trials_file)
        shutil.move(temporary, trials_fname)

    return trials
