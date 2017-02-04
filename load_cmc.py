def load_cmc():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv

    objects, labels = [None, None]
    with open('data/cmc.data', 'r') as f:
        data = read_csv(f, header=None)
        objects = data.iloc[:, range(9)].values.tolist()
        labels = data.iloc[:, [9]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    objects, labels = shuffle(objects, labels)

    return objects.tolist(), labels.tolist()
