from copy import deepcopy as dc
from tree import generate_random_tree, crossover_trees, tree_mutation
from random import random, choice, sample
from time import time

import numpy as np


class AdaSS:
    def __init__(self, **kwargs):
        self.pool = dc(kwargs['pool'])
        self.tree = None

    def fit(self, max_init_depth, max_iterations, decay_time, f_mut, f_co, f_el, n_trees, train, validation, **kwargs):
        start_time = time()
        max_init_depth = max_init_depth
        max_iterations = max_iterations
        decay_time = decay_time
        f_mut = f_mut
        f_co = f_co
        f_el = f_el
        n_trees = n_trees

        if f_mut + f_co + f_el != 1:
            raise Exception('Factors do not sum to 1')

        train = train
        validation = validation

        def evaluate_tree(tree, type):
            if type == 'train':
                labels = np.transpose([classifier.predict(train[0]) for classifier in self.pool])
                weights = tree.eval(train[0])
                classes = np.max(labels)
                categorical_labels = np.zeros((len(train[0]), classes + 1)).tolist()
                for object_index in range(len(train[0])):
                    for classifier_index in range(len(self.pool)):
                        categorical_labels[object_index][labels[object_index][classifier_index]] += \
                            weights[object_index][classifier_index]
                decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

                return sum([1 if decisions[i] == train[1][i] else 0 for i in range(len(train[1]))]) * 1. / len(train[0])


            elif type == 'validation':
                labels = np.transpose([classifier.predict(validation[0]) for classifier in self.pool])
                weights = tree.eval(validation[0])
                classes = np.max(labels)
                categorical_labels = np.zeros((len(validation[0]), classes + 1)).tolist()
                for object_index in range(len(validation[0])):
                    for classifier_index in range(len(self.pool)):
                        categorical_labels[object_index][labels[object_index][classifier_index]] += \
                            weights[object_index][classifier_index]
                decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

                return sum(
                    [1 if decisions[i] == validation[1][i] else 0 for i in range(len(validation[1]))]) * 1. / len(
                    validation[0])

            else:
                raise Exception('Bad type %s; expected train or validation' % type)

        print('Initialization completed, \ttime: %f' % (time() - start_time))

        population = [generate_random_tree(len(train[0][0]),
                                           lambda: [random() for i in range(len(self.pool))],
                                           max_init_depth) for i in range(n_trees)]
        elite = None
        best_in_cycle = [(0, 0)]
        no_improvement_cycles = 0
        print('Beginning evolutionary loop, \ttime: %f' % (time() - start_time))

        for cycle in range(max_iterations + 1):
            print('Generation %d' % cycle)
            if cycle > 0:
                new_generation = []
                mutation = sample(range(len(population)), int(len(population) * f_mut))
                crossover = sample(range(len(population)), int(len(population) * f_co))

                for tree_index in mutation:
                    new_generation.append(tree_mutation(population[tree_index],
                                                        len(train[0][0]),
                                                        lambda: [random() for i in range(len(self.pool))],
                                                        f1=0.55,
                                                        f2=0.35,
                                                        f3=0.1))

                for tree_index in crossover:
                    new_generation.append(crossover_trees(population[tree_index],
                                                          choice(population)))

                population = dc(new_generation) + dc(elite)

            tree_sizes = [tree.check_descendants() for tree in population]
            print('Tree size \tMin: %d\tMax: %d\tAverage: %0.1f' %
                  (min(tree_sizes), max(tree_sizes), np.mean(tree_sizes)))
            accuracy_train = sorted([(evaluate_tree(tree, 'train'), tree_index)
                                     for tree_index, tree in enumerate(population)], key=lambda x: x[0])[::-1]
            accuracy_validation = sorted([(evaluate_tree(tree, 'validation'), tree_index)
                                          for tree_index, tree in enumerate(population)], key=lambda x: x[0])[::-1]
            elite = [dc(population[tree_index[1]]) for tree_index in accuracy_train[:int(n_trees * f_el)]]
            best_in_cycle.append((accuracy_train[0][0], accuracy_validation[0][0]))
            print('Best in cycle trees: %0.4f, %0.4f, \ttime: %0.2f' % (*best_in_cycle[-1], time() - start_time))

            no_improvement_cycles = no_improvement_cycles + 1 \
                if best_in_cycle[-1][1] <= sorted(best_in_cycle[:-1], key=lambda x: x[1])[-1][1] \
                else 0
            print(('\033[93mNo improvement\033[0m' if no_improvement_cycles > 0 else '\033[92mImprovement\033[0m'))
            if no_improvement_cycles == decay_time:
                self.tree = population[accuracy_train[0][1]]
                break

        return best_in_cycle[1:]

    def predict(self, test, **kwargs):
        labels = np.transpose([classifier.predict(test) for classifier in self.pool])
        weights = self.tree.eval(test)
        classes = np.max(labels)
        categorical_labels = np.zeros((len(test), classes + 1)).tolist()
        for object_index in range(len(test)):
            for classifier_index in range(len(self.pool)):
                categorical_labels[object_index][labels[object_index][classifier_index]] += weights[object_index][
                    classifier_index]
        decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

        return decisions
