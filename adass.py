from copy import deepcopy as dc
from tree import generate_random_tree, crossover_trees, tree_mutation
from random import random

import numpy as np

class AdaSS:
    def __init__(self, **kwargs):
        self.pool = dc(kwargs['pool'])
        self.tree = None

    def fit(self, **kwargs):
        max_init_depth = kwargs['max_init_depth']
        f_mut = kwargs['f_mut']
        f_co = kwargs['f_co']
        f_el = kwargs['f_el']
        n_trees = kwargs['n_trees']

        if f_mut + f_co + f_el != 1:
            raise Exception('Factors do not sum to 1')

        train = kwargs['train']
        validation = kwargs['validation']

        population = [generate_random_tree(len(train[0]),
                                           lambda: [random() for i in range(len(self.pool))],
                                           max_init_depth) for i in range(n_trees)]
        accuracy_train = evaluate_
        accuracy_validation =
        elite =

    def predict(self, **kwargs):
        pass
