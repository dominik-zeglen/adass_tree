class BootstrapPool:
    def __init__(self, base_classifier, n_classifiers, *args, **kwargs):
        if 'random_state' in kwargs:
            self.seed = kwargs['random_state']

        self.pool = [base_classifier(*args, **kwargs) for i in range(n_classifiers)]
        self.seed = None

    def fit(self, train, labels):
        from random import choice, seed

        if self.seed:
            seed(self.seed)

        for classifier in self.pool:
            indexes = [choice(range(len(labels))) for i in range(len(labels))]

            X = [train[index] for index in indexes]
            y = [labels[index] for index in indexes]

            classifier.fit(X, y)

        return self

    def get_classifiers(self):
        return self.pool


class Pool:
    def __init__(self, base_classifier, n_classifiers, *args, **kwargs):
        if 'random_state' in kwargs:
            self.seed = kwargs['random_state']

        self.pool = [base_classifier(*args, **kwargs) for i in range(n_classifiers)]

    def fit(self, train, labels):
        for classifier in self.pool:
            classifier.fit(train, labels)

        return self

    def get_classifiers(self):
        return self.pool
