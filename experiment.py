import numpy as np
import matplotlib.pyplot as plt

from random import choice, seed, random
from adass import AdaSS
from load_cmc import load_cmc
from pool_methods import BootstrapPool
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


def update_progress(progress):
    n = 10
    print('\r[{0}{1}] {2}%'.format('#'*progress, '-'*(n - progress), progress * n), end='')

random_seed = 2229
seed(random_seed)


objects, labels, num_features, num_classes = load_cmc()
pool = BootstrapPool(DecisionTreeClassifier, 30, random_state=random_seed).fit(
    *np.transpose([choice(np.array([objects, labels]).T) for i in range(int(len(labels) / 2))])
).get_classifiers()
scores = []

iterator = 0
kfold5 = KFold(n_splits=5, random_state=random_seed)
for learn_indexes, test_indexes in kfold5.split(objects, labels):
    learn = [[objects[index] for index in learn_indexes], [labels[index] for index in learn_indexes]]
    test = [[objects[index] for index in test_indexes], [labels[index] for index in test_indexes]]

    kfold2 = KFold(n_splits=2, random_state=random_seed)
    for train_indexes, val_indexes in kfold2.split(learn[0]):
        update_progress(iterator)
        iterator += 1
        train = [[learn[0][index] for index in train_indexes], [learn[1][index] for index in train_indexes]]
        validation = [[learn[0][index] for index in val_indexes], [learn[1][index] for index in val_indexes]]

        for classifier in pool:
            scores.append(sum(
                [1. if classifier.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
                test[1]))

        adass = AdaSS(pool=pool)
        efficiency = adass.fit(train=train,
                               validation=validation,
                               max_init_depth=12,
                               decay_time=3,
                               max_iterations=20,
                               f_co=0.65,
                               f_mut=0.2,
                               f_el=0.15,
                               n_trees=40)

        efficiency = np.transpose(efficiency)

        plt.subplot(2, 5, iterator)
        plt.plot(range(len(efficiency[0])), efficiency[0], 'b', range(len(efficiency[1])), efficiency[1], 'r')
        for score in scores[-len(pool):]:
            plt.axhline(score, color='g', linestyle='dashed')

        scores.append(
            sum([1. if adass.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
                test[1]))
        plt.axhline(scores[-1], color='g')

        update_progress(iterator)

plt.show()
plt.boxplot(np.array(scores).reshape((10, len(pool) + 1)))
plt.show()

means = np.mean(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1)
medians = np.median(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1)
stds = np.std(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1)

print('\nMean accuracy:\t\t\t Min: %0.3f\t Max: %0.3f\t Average: %0.3f\t AdaSS: %0.3f' %
      (min(means[:-1]), max(means[:-1]), np.mean(means[:-1], axis=0), means[-1]))
print('Median accuracy:\t\t Min: %0.3f\t Max: %0.3f\t Average: %0.3f\t AdaSS: %0.3f' %
      (min(medians[:-1]), max(medians[:-1]), np.mean(medians[:-1], axis=0), medians[-1]))
print('Standard deviations:\t Min: %0.3f\t Max: %0.3f\t Average: %0.3f\t AdaSS: %0.3f' %
      (min(stds[:-1]), max(stds[:-1]), np.mean(stds[:-1], axis=0), stds[-1]))
