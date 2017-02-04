class AdaSS:
    def __init__(self, **kwargs):
        from copy import deepcopy as dc

        self.pool = dc(kwargs['pool'])
        self.tree = None

    def fit(self, max_init_depth, max_iterations, decay_time, f_mut, f_co, f_el, n_trees, train, validation,
            max_clusters, **kwargs):
        from copy import deepcopy as dc
        from adass_weightarray_helpers import generate_random_individual, crossover_individuals, individual_mutation
        from random import random, choice, sample, seed
        from time import time

        import numpy as np

        start_time = time()
        if 'random_state' in kwargs:
            seed(kwargs['random_state'])

        if f_mut + f_co + f_el != 1:
            raise Exception('Factors do not sum to 1')

        train = train
        validation = validation

        def evaluate_individual(individual, type):
            if type == 'train':
                labels = np.transpose([classifier.predict(train[0]) for classifier in self.pool])
                weights = individual.eval(train[0])
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
                weights = individual.eval(validation[0])
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

        print('Initialization completed, \ttime: %0.2f' % (time() - start_time))

        population = [generate_random_individual(len(train[0][0]),
                                                 len(self.pool),
                                                 lambda: int(random() * max_clusters),
                                                 max_clusters,
                                                 max_init_depth) for i in range(n_trees)]
        elite = None
        best_in_cycle = [(0, 0)]
        no_improvement_cycles = 0
        print('Beginning evolutionary loop, \ttime: %0.2f' % (time() - start_time))

        for cycle in range(max_iterations + 1):
            print('Generation %d' % cycle)
            if cycle > 0:
                new_generation = []
                mutation = sample(range(len(population)), int(len(population) * f_mut))
                crossover = sample(range(len(population)), int(len(population) * f_co))

                for tree_index in mutation:
                    new_generation.append(individual_mutation(population[tree_index],
                                                              len(train[0][0]),
                                                              len(self.pool),
                                                              lambda: int(random() * max_clusters),
                                                              f1=0.8 * (1 - cycle / max_iterations),
                                                              f2=0.2,
                                                              f3=1 - cycle / max_iterations))

                for tree_index in crossover:
                    new_generation.append(crossover_individuals(population[tree_index],
                                                                choice(population)))

                population = dc(new_generation) + dc(elite)

            tree_sizes = [ind.tree.check_descendants() + 1 for ind in population]
            print('Population size:\t%d' % len(population))
            print('Tree size \tMin: %d\tMax: %d\tAverage: %0.1f' %
                  (min(tree_sizes), max(tree_sizes), np.mean(tree_sizes)))
            accuracy_train = sorted([(evaluate_individual(ind, 'train'), ind_index)
                                     for ind_index, ind in enumerate(population)], key=lambda x: x[0])[::-1]
            accuracy_validation = sorted([(evaluate_individual(ind, 'validation'), ind_index)
                                          for ind_index, ind in enumerate(population)], key=lambda x: x[0])[::-1]
            elite = [dc(population[ind_index[1]]) for ind_index in accuracy_train[:int(n_trees * f_el)]]
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
        import numpy as np

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
