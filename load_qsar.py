def load_qsar():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv

    objects, labels = [None, None]
    with open('data/qsar.csv', 'r') as f:
        data = read_csv(f, header=None, delimiter=';')
        objects = data.iloc[:, range(41)].values.tolist()
        labels = data.iloc[:, [41]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    objects, labels = shuffle(objects, labels)

    return objects.tolist(), labels.tolist()
