class MajorityVoting:
    def __init__(self, pool):
        from copy import deepcopy as dc

        self.pool = dc(pool)

    def predict(self, X):
        from collections import Counter
        from numpy import transpose

        decisions = transpose([classifier.predict(X) for classifier in self.pool])
        return [Counter(decisions[i]).most_common(1)[0][0] for i in range(len(decisions))]