class Individual():
    def __init__(self):
        self.tree = None
        self.weight_array = None

    def _eval(self, object):
        return self.weight_array[self.tree._eval(object)]

    def eval(self, object_list):
        return [self._eval(object) for object in object_list]


def generate_random_individual(features, n_pool, terminal_node_callback, max_clusters, max_depth=10):
    from tree import generate_random_tree
    from random import random

    individual = Individual()
    individual.tree = generate_random_tree(features, terminal_node_callback, max_depth)
    individual.weight_array = [[random() for j in range(n_pool)] for i in range(max_clusters)]

    return individual


def crossover_individuals(ind1, ind2):
    from tree import crossover_trees
    from random import random

    individual = Individual()
    individual.tree = crossover_trees(ind1.tree, ind2.tree)
    co_point = int(random() * len(ind1.weight_array))
    individual.weight_array = [ind1.weight_array[i] if i < co_point else ind2.weight_array[i]
                               for i in range(len(ind1.weight_array))]

    return individual


def individual_mutation(individual, features, n_pool, terminal_node_callback, f1=0.33, f2=0.33, f3=0.5):
    from tree import tree_mutation
    from random import random

    ind = individual
    roll = random()

    if roll > f3:
        # Weight array mutation
        ind.weight_array[int(random() * len(ind.weight_array))] = [random() for i in range(n_pool)]

    else:
        # Tree mutation
        ind.tree = tree_mutation(ind.tree, features, terminal_node_callback, f1, f2)

    return ind
