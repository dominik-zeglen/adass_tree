import numpy as np
import load_dataset

from adass import AdaSS as AdaSSTree
from adass_old import AdaSS as AdaSSCluster
from clustering_and_selection import CS
from majority_voting import MajorityVoting
from pool_methods import BootstrapPool
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier as DTC
from pandas import DataFrame, read_pickle
from os import makedirs
from random import choice

params = [3, 6, 9, 12]
classifier_labels = [*['AdaSS (tree) - init depth: %d' % param for param in params],
                     *['AdaSS (centroid) - clusters: %d' % param for param in params],
                     *['Clustering and Selection - clusters: %d' % param for param in params],
                     'Majority Voting']

for name, dataset in load_dataset.__dict__.items():
    if callable(dataset):
        objects, labels, num_features, num_classes = dataset()
        kfold = KFold(n_splits=10)

        for fold, fold_data in enumerate(kfold.split(objects, labels)):
            learn_indexes, test_indexes = fold_data
            learn = [[objects[index] for index in learn_indexes], [labels[index] for index in learn_indexes]]
            test = [[objects[index] for index in test_indexes], [labels[index] for index in test_indexes]]

            pool = BootstrapPool(DTC, 30).fit(
                *np.transpose([choice(np.array([objects, labels]).T) for i in range(int(len(labels) / 2))])
            ).get_classifiers()

            classifiers = [*[(
                                 AdaSSTree,
                                 {
                                     'pool': pool
                                 },
                                 {
                                     'max_init_depth': param,
                                     'max_iterations': 10,
                                     'decay_time': 3,
                                     'f_co': 0.65,
                                     'f_mut': 0.2,
                                     'f_el': 0.15,
                                     'n_trees': 60
                                 }
                             ) for param in params],

                           *[(
                                 AdaSSCluster,
                                 {
                                     'pool': pool
                                 },
                                 {
                                     'n_clusters': param,
                                     'max_iterations': 10,
                                     'decay_time': 3,
                                     'f_co': 0.65,
                                     'f_mut': 0.2,
                                     'f_el': 0.15,
                                     'n_population': 60,
                                     'mut_range': 0.25
                                 }
                             ) for param in params],
                           *[(
                                 CS,
                                 {
                                     'pool': pool,
                                     'n_clusters': param
                                 },
                                 {

                                 }
                             ) for param in params],
                           (
                               MajorityVoting,
                               {
                                   'pool': pool,
                               },
                               {

                               }
                           )
                           ]

            for classifier_index, classifier_data in enumerate(classifiers):
                classifier_type, kwargs1, kwargs2 = classifier_data
                classifier = None

                if kwargs1:
                    classifier = classifier_type(**kwargs1)
                else:
                    classifier = classifier_type()

                if classifier.__class__.__name__ != 'MajorityVoting':
                    if kwargs2:
                        classifier.fit(*learn, **kwargs2)
                    else:
                        classifier.fit(*learn)

                path = 'pickles/%s/%s' % (name[5:], classifier_index)
                print('\rSaving %s/%s.pickle' % (path, fold), end='')
                makedirs(path, exist_ok=True)
                decisions = DataFrame(np.transpose([classifier.predict(test[0]), test[1]])). \
                    to_pickle('%s/%s.pickle' % (path, fold))

for name, dataset in load_dataset.__dict__.items():
    if callable(dataset):
        print('')
        for i in range(3 * len(params) + 1):
            scores = []
            for j in range(10):
                df = read_pickle('pickles/%s/%s/%s.pickle' % (name[5:], i, j))
                decisions = np.asarray(df[0].values.tolist())
                labels = np.asarray(df[1].values.tolist())

                scores.append(np.mean(np.equal(decisions, labels)))

            print('%s: %0.3f +/- %0.3f [%s]' % (name[5:], np.mean(scores), np.std(scores), classifier_labels[i]))
