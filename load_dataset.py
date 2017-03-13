def load_cmc():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv
    from numpy import ravel

    objects, labels = [None, None]
    with open('data/cmc.data', 'r') as f:
        data = read_csv(f, header=None)
        objects = data.iloc[:, range(9)].values.tolist()
        labels = data.iloc[:, [9]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(ravel(labels))
    objects, labels = shuffle(objects, labels)

    return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)


def load_parkinsons():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv
    from numpy import ravel

    objects, labels = [None, None]
    with open('data/parkinsons.data', 'r') as f:
        data = read_csv(f)
        objects = data.iloc[:, [*range(1, 17), *range(18, 23)]].astype('float64').values.tolist()
        labels = data.iloc[:, [17]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(ravel(labels))
    objects, labels = shuffle(objects, labels)

    return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)


def load_qsar():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv
    from numpy import ravel

    objects, labels = [None, None]
    with open('data/qsar.csv', 'r') as f:
        data = read_csv(f, header=None, delimiter=';')
        objects = data.iloc[:, range(41)].astype('float64').values.tolist()
        labels = data.iloc[:, [41]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(ravel(labels))
    objects, labels = shuffle(objects, labels)

    return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)


def load_yeast():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv
    from numpy import ravel

    objects, labels = [None, None]
    with open('data/yeast.data', 'r') as f:
        data = read_csv(f, header=None, delimiter=r"\s+")
        objects = data.iloc[:, [*range(1, 9)]].astype('float64').values.tolist()
        labels = data.iloc[:, [9]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(ravel(labels))
    objects, labels = shuffle(objects, labels)

    return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)