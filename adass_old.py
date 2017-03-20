from copy import deepcopy as dc
from random import random, choice, sample, gauss
from time import time

import numpy as np
import matplotlib.pyplot as plt


class AdaSS:
    def __init__(self, **kwargs):
        self.pool = dc(kwargs['pool'])
        self.model = None

    def fit(self, n_clusters, max_iterations, decay_time, f_mut, f_co, f_el,
            n_population, mut_range, learn, verbose=False, **kwargs):
        from collections import Counter
        from sklearn.model_selection import ShuffleSplit
        from numpy.linalg import norm

        start_time = time()

        if f_mut + f_co + f_el != 1:
            raise Exception('Factors do not sum to 1')

        learn = np.random.permutation(np.transpose(learn))
        train = np.transpose(learn[:int(len(learn) * 2 / 3)]).tolist()
        validation = np.transpose(learn[int(len(learn) * 2 / 3):]).tolist()
        num_features = len(train[0][0])
        num_classes = len(Counter(train[1]))
        num_t_obs = len(train[0])
        num_v_obs = len(validation[0])

        def evaluate_chromosome(c, w, type):
            if type == 'train':
                labels = np.transpose([classifier.predict(train[0]) for classifier in self.pool])
                distances = np.asarray([np.asarray(train[0]) - centroid for centroid in c])
                distances = np.transpose(np.sum(distances * distances, axis=2)).tolist()
                weights = [w[obs.index(min(obs))] for obs in distances]
                categorical_labels = np.zeros((num_t_obs, num_classes + 1)).tolist()
                for object_index in range(num_t_obs):
                    for classifier_index in range(len(self.pool)):
                        categorical_labels[object_index][labels[object_index][classifier_index]] += \
                            weights[object_index][classifier_index]
                decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

                return sum([1 if decisions[i] == train[1][i] else 0 for i in range(num_t_obs)]) * 1. / num_t_obs

            elif type == 'validation':
                labels = np.transpose([classifier.predict(validation[0]) for classifier in self.pool])
                distances = np.asarray([np.asarray(validation[0]) - centroid for centroid in c])
                distances = np.transpose(np.sum(distances * distances, axis=2)).tolist()
                weights = [w[obs.index(min(obs))] for obs in distances]
                categorical_labels = np.zeros((num_v_obs, num_classes + 1)).tolist()
                for object_index in range(num_v_obs):
                    for classifier_index in range(len(self.pool)):
                        categorical_labels[object_index][labels[object_index][classifier_index]] += \
                            weights[object_index][classifier_index]
                decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

                return sum([1 if decisions[i] == validation[1][i] else 0 for i in range(num_v_obs)]) * 1. / num_v_obs

            else:
                raise Exception('Bad type %s; expected train or validation' % type)

        def mt(ind, m_range, f):
            c = np.array(ind[0][:])
            w = np.array(ind[1][:])

            if random() > f:
                c[int(random() * len(c))] += np.array([gauss(0, m_range) for i in range(len(c[0]))])
            else:
                w[int(random() * len(w))] += np.array([gauss(0, m_range) for i in range(len(w[0]))])

            return (c, w)

        def cv(ind1, ind2):
            point = int(random() * len(ind1[0][0]))
            c = np.hstack([np.array(ind1[0])[:, point:], np.array(ind2[0])[:, :point]]).tolist()
            w = np.hstack([np.array(ind1[1])[:, point:], np.array(ind2[1])[:, :point]]).tolist()

            return (c, w)

        if verbose:
            print('Initialization completed, \ttime: %0.2f' % (time() - start_time))

        population = [[[[random() * 2 - 1 for j in range(num_features)] for k in range(n_clusters)],
                       [[random() for j in range(len(self.pool))] for k in range(n_clusters)]
                       ] for i in range(n_population)]
        elite = None
        best_in_cycle = [(0, 0)]
        no_improvement_cycles = 0
        if verbose:
            print('Beginning evolutionary loop, \ttime: %0.2f' % (time() - start_time))

        for cycle in range(max_iterations + 1):
            if verbose:
                print('Generation %d' % cycle)
            if cycle > 0:
                new_generation = []
                mutation = dc(sample(range(len(population)), int(len(population) * f_mut)))
                crossover = dc(sample(range(len(population)), int(len(population) * f_co)))

                for chr_index in mutation:
                    new_generation.append(mt(population[chr_index], mut_range, cycle / max_iterations))

                for chr_index in crossover:
                    new_generation.append(cv(population[chr_index], population[choice(crossover)]))

                population = dc(new_generation) + dc(elite)

            accuracy_train = sorted([(evaluate_chromosome(chr[0], chr[1], 'train'), chr_index)
                                     for chr_index, chr in enumerate(population)], key=lambda x: x[0])[::-1]
            accuracy_validation = sorted([(evaluate_chromosome(chr[0], chr[1], 'validation'), chr_index)
                                          for chr_index, chr in enumerate(population)], key=lambda x: x[0])[::-1]
            elite = [dc(population[chr_index[1]]) for chr_index in accuracy_train[:int(n_clusters * f_el)]]
            best_in_cycle.append((accuracy_train[0][0], accuracy_validation[0][0]))
            if verbose:
                print('Best in cycle: %0.4f, %0.4f, \ttime: %0.2f' % (*best_in_cycle[-1], time() - start_time))

            no_improvement_cycles = no_improvement_cycles + 1 \
                if best_in_cycle[-1][1] <= sorted(best_in_cycle[:-1], key=lambda x: x[1])[-1][1] \
                else 0
            if verbose:
                print(('\033[93mNo improvement\033[0m' if no_improvement_cycles > 0 else '\033[92mImprovement\033[0m'))
            if no_improvement_cycles == decay_time or cycle == (max_iterations - 1):
                self.model = dc(population[accuracy_train[0][1]])
                break

        return best_in_cycle[1:]

    def predict(self, test, **kwargs):
        labels = np.transpose([classifier.predict(test) for classifier in self.pool])
        distances = np.asarray([np.asarray(test) - centroid for centroid in self.model[0]])
        distances = np.transpose(np.sum(distances * distances, axis=2)).tolist()
        weights = [self.model[1][obs.index(min(obs))] for obs in distances]
        classes = np.max(labels)
        categorical_labels = np.zeros((len(test), classes + 1)).tolist()
        for object_index in range(len(test)):
            for classifier_index in range(len(self.pool)):
                categorical_labels[object_index][labels[object_index][classifier_index]] += weights[object_index][
                    classifier_index]
        decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

        return decisions
