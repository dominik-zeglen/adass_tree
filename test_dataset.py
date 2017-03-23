import numpy as np
import load_dataset

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold


print("Total number of datasets: %d" % sum(1 if callable(item[1]) else 0 for item in load_dataset.__dict__.items()))

for name, dataset in load_dataset.__dict__.items():
    if callable(dataset):
        objects, labels, num_features, num_classes = dataset()
        scores = []

        kfold = KFold(n_splits=10)

        for learn_indexes, test_indexes in kfold.split(objects, labels):
            learn = [[objects[index] for index in learn_indexes], [labels[index] for index in learn_indexes]]
            test = [[objects[index] for index in test_indexes], [labels[index] for index in test_indexes]]

            lda = LDA()
            lda.fit(*learn)

            scores.append(
                sum([1. if lda.predict([test[0][i]])[0] == test[1][i] else 0. for i in range(len(test[1]))]) / len(
                    test[1]))

        print("%s\n\tclasses: %d\n\tobservations: %d\n\tfeatures: %d\n\tlda score: %f " %
              (
                  name[5:],
                  num_classes,
                  len(labels),
                  num_features,
                  np.mean(scores)
              ))
