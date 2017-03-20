import numpy as np
import matplotlib.pyplot as plt

from random import choice, seed
from time import time
from adass import AdaSS
from adass_old import AdaSS as AdaSSDensity
from load_dataset import *
from pool_methods import BootstrapPool
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


def update_progress(progress):
    n = 20
    print('\r[{0}{1}] {2}%'.format('#' * progress, '-' * (n - progress), 100 * progress / n), end='')


random_seed = 2229
seed(random_seed)

objects, labels, num_features, num_classes = load_cmc()
pool = BootstrapPool(DecisionTreeClassifier, 30, random_state=random_seed).fit(
    *np.transpose([choice(np.array([objects, labels]).T) for i in range(int(len(labels) / 2))])
).get_classifiers()
scores = []
times = []

iterator = 0
kfold5 = KFold(n_splits=10, random_state=random_seed)
for learn_indexes, test_indexes in kfold5.split(objects, labels):
    learn = [[objects[index] for index in learn_indexes], [labels[index] for index in learn_indexes]]
    test = [[objects[index] for index in test_indexes], [labels[index] for index in test_indexes]]

    update_progress(iterator)

    for classifier in pool:
        scores.append(sum(
            [1. if classifier.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
            test[1]))

    sgdensemble = SGDEnsembleLearner(pool=pool)
    sgdensemble.fit(learn=learn,
                    learning_rate=0.01,
                    max_iterations=100)

    scores.append(
        sum([1. if sgdensemble.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
            test[1]))

    adass = AdaSS(pool=pool)
    start_time = time()
    adass.fit(learn=learn,
              max_init_depth=12,
              decay_time=3,
              max_iterations=20,
              f_co=0.65,
              f_mut=0.2,
              f_el=0.15,
              n_trees=100)
    times.append(time() - start_time)

    iterator += 1
    update_progress(iterator)

    scores.append(
        sum([1. if adass.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
            test[1]))

    adass_density = AdaSSDensity(pool=pool)
    start_time = time()
    adass_density.fit(learn=learn,
                      n_clusters=12,
                      decay_time=3,
                      max_iterations=20,
                      f_co=0.65,
                      f_mut=0.2,
                      f_el=0.15,
                      n_population=100,
                      mut_range=0.2)
    times.append(time() - start_time)

    scores.append(
        sum([1. if adass_density.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
            test[1]))

    iterator += 1
    update_progress(iterator)

plt.boxplot(np.array(scores).reshape((10, len(pool) + 2)) * 100,
            labels=[*['C{0}'.format(i + 1) for i in range(len(pool))], 'AdaSS', 'Old AdaSS'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy [%]')
plt.show()

means = np.mean(np.array(scores).reshape((10, len(pool) + 2)).T, axis=1) * 100
medians = np.median(np.array(scores).reshape((10, len(pool) + 2)).T, axis=1) * 100
stds = np.std(np.array(scores).reshape((10, len(pool) + 2)).T, axis=1) * 100
times = np.mean(np.array(times).reshape((10, 2)).T, axis=1)

print('\nMean accuracy:\t\t\t Min: %0.3f\t Max: %0.3f\t Average: %0.3f\t AdaSS: %0.3f\t Old AdaSS: %0.3f' %
      (min(means[:-2]), max(means[:-2]), np.mean(means[:-2], axis=0), means[-2], means[-1]))
print('Median accuracy:\t\t Min: %0.3f\t Max: %0.3f\t Average: %0.3f\t AdaSS: %0.3f\t Old AdaSS: %0.3f' %
      (min(medians[:-2]), max(medians[:-2]), np.mean(medians[:-2], axis=0), medians[-2], medians[-1]))
print('Standard deviations:\t Min: %0.3f\t\t Max: %0.3f\t\t Average: %0.3f\t\t AdaSS: %0.3f\t Old AdaSS: %0.3f' %
      (min(stds[:-2]), max(stds[:-2]), np.mean(stds[:-2], axis=0), stds[-2], stds[-1]))
print('Time efficiency comparison:\t AdaSS: %0.3f\t\t Old AdaSS: %0.3f' % (times[0], times[1]))