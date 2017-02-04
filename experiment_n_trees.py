import numpy as np
import matplotlib.pyplot as plt

from random import choice, seed
from adass import AdaSS
from load_qsar import load_qsar
from pool_methods import BootstrapPool
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

random_seed = 2229
seed(random_seed)
objects, labels = load_qsar()
scores = []
ensemble_scores = []
test_range = [*range(2, 50, 2)]

for n_ensemble in test_range:
    iterator = 0
    kfold5 = KFold(n_splits=5, random_state=random_seed)
    pool = BootstrapPool(DecisionTreeClassifier, n_ensemble).fit(
        *np.transpose([choice(np.array([objects, labels]).T) for i in range(int(len(labels) / 4))])
    ).get_classifiers()

    for learn_indexes, test_indexes in kfold5.split(objects, labels):
        learn = [[objects[index] for index in learn_indexes], [labels[index] for index in learn_indexes]]
        test = [[objects[index] for index in test_indexes], [labels[index] for index in test_indexes]]

        kfold2 = KFold(n_splits=2, random_state=random_seed)
        for train_indexes, val_indexes in kfold2.split(learn[0]):
            iterator += 1
            train = [[learn[0][index] for index in train_indexes], [learn[1][index] for index in train_indexes]]
            validation = [[learn[0][index] for index in val_indexes], [learn[1][index] for index in val_indexes]]

            for classifier in pool:
                ensemble_scores.append(sum(
                    [1. if classifier.predict([test[0][i]])[0] == test[1][i] else 0. for i in
                     range(len(test[1]))]) / len(
                    test[1]))

            adass = AdaSS(pool=pool)
            adass.fit(train=train,
                      validation=validation,
                      max_init_depth=8,
                      decay_time=5,
                      max_iterations=15,
                      f_co=0.75,
                      f_mut=0.15,
                      f_el=0.1,
                      n_trees=20)

            scores.append(
                sum([1. if adass.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
                    test[1]))

# plt.show()
# plt.boxplot(np.array(scores).reshape((10, len(pool) + 1)))
# plt.show()
# print('\nAverage accuracy:\t\t\t%s' % np.mean(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1))
# print('\nMedian accuracy:\t\t\t%s' % np.median(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1))
# print('\nStandard deviations:\t%s' % np.std(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1))

scores = np.array(scores).reshape((len(test_range), 10)).T
plt.boxplot(scores)
plt.show()