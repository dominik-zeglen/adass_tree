class SGDEnsembleLearner:
    def __init__(self, pool):
        from copy import deepcopy as dc
        from random import random
        from numpy import asarray

        self.pool = dc(pool)
        self.weights = asarray([.5 for i in range(len(self.pool))]).astype('float64')

    def fit(self, learn, learning_rate, max_iterations, batch_size=1, verbose=False, output_progress=False):
        from numpy import asarray, equal, mean
        from sklearn.utils import shuffle

        output = []
        for i in range(max_iterations):
            objects, labels = shuffle(*learn)
            for j in range(0, len(learn), batch_size):
                decisions = asarray([classifier.predict(objects[j:(j + batch_size)]) for classifier in self.pool]).T
                self.weights += mean(equal(decisions, asarray(labels[j:(j + batch_size)]).reshape(batch_size, 1)).T * 2 - 1, axis=1) * learning_rate

            if verbose:
                print("Iteration %d, accuracy: %0.3f" %
                      (i, sum(equal(labels, self.predict(objects))) * 100. / len(objects)))

            if output_progress:
                output.append(sum(equal(labels, self.predict(objects))) * 100. / int(len(labels)))

        if output_progress:
            return output
        else:
            return self

    def predict(self, test):
        from numpy import transpose, asarray, sum, max, zeros
        labels = transpose([classifier.predict(test) for classifier in self.pool])
        classes = max(labels)
        categorical_labels = zeros((len(test), classes + 1)).tolist()
        for object_index in range(len(test)):
            for classifier_index in range(len(self.pool)):
                categorical_labels[object_index][labels[object_index][classifier_index]] += self.weights[classifier_index]
        decisions = [label_list.index(max(label_list)) for label_list in categorical_labels]

        return decisions