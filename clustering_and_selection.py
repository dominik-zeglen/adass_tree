class CS:
    def __init__(self, pool, n_clusters):
        from copy import deepcopy as dc
        from sklearn.cluster import KMeans

        self.pool = dc(pool)
        self.cluster = KMeans(n_clusters=n_clusters)
        self.best_classifiers = [0 for i in range(n_clusters)]

    def fit(self, X, y):
        from numpy import transpose, equal, sum

        self.cluster = self.cluster.fit(X)
        cluster_labels = self.cluster.predict(X)

        for cluster_index in range(len(self.cluster.cluster_centers_)):
            objects, labels = transpose([(X[i], y[i]) for i in range(len(y)) if cluster_labels[i] == cluster_index])
            scores = []

            for classifier in self.pool:
                scores.append(sum(equal(classifier.predict(objects.tolist()), labels)))

            self.best_classifiers[cluster_index] = scores.index(max(scores))

    def predict(self, X):
        from numpy import asarray
        return [self.pool[self.best_classifiers[self.cluster.predict(asarray(x).reshape(1, -1))[0]]].predict(asarray(x).reshape(1, -1))[0] for x in X]