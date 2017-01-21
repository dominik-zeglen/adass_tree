import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from adass import AdaSS
from load_cmd import load_cmc
from load_parkinsons import load_parkinsons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold

objects, labels = load_parkinsons()

base = DecisionTreeClassifier
pool = [base().fit(objects[:60], labels[:60]),
        base().fit(objects[60:120], labels[60:120]),
        base().fit(objects[120:180], labels[120:180])]
scores = []

# train[0], validation[0], train[1], validation[1] = train_test_split(objects, labels, test_size=0.2, random_state=5)

kfold5 = KFold(n_splits=5)
for learn_indexes, test_indexes in kfold5.split(objects, labels):
    learn = [[objects[index] for index in learn_indexes], [labels[index] for index in learn_indexes]]
    test = [[objects[index] for index in test_indexes], [labels[index] for index in test_indexes]]

    kfold2 = KFold(n_splits=2)
    for train_indexes, val_indexes in kfold2.split(learn[0]):
        train = [[learn[0][index] for index in train_indexes], [learn[1][index] for index in train_indexes]]
        validation = [[learn[0][index] for index in val_indexes], [learn[1][index] for index in val_indexes]]

        for classifier in pool:
            scores.append(sum(
                [1. if classifier.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
                test[1]))

        adass = AdaSS(pool=pool)
        efficiency = adass.fit(train=train,
                               validation=validation,
                               max_init_depth=10,
                               decay_time=5,
                               max_iterations=200,
                               f_co=0.55,
                               f_mut=0.4,
                               f_el=0.05,
                               n_trees=100)

        efficiency = np.transpose(efficiency)

        # plt.plot(range(len(efficiency[0])), efficiency[0], 'b', range(len(efficiency[1])), efficiency[1], 'r')
        # for score in scores[-3:]:
        #     plt.axhline(score, color='g', linestyle='dashed')

        scores.append(
            sum([1. if adass.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
                test[1]))
        # plt.axhline(scores[-1], color='g')
        # plt.show()

print('\nFinal scores:\t\t\t%s' % np.mean(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1))
print('Standard deviations:\t%s' % np.std(np.array(scores).reshape((10, len(pool) + 1)).T, axis=1))
