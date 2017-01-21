def load_parkinsons():
    from sklearn.utils import shuffle
    from sklearn.preprocessing import Normalizer, LabelEncoder
    from pandas import read_csv

    objects, labels = [None, None]
    with open('data/parkinsons.data', 'r') as f:
        data = read_csv(f)
        objects = data.iloc[:, [*range(1, 17), *range(18, 23)]].values.tolist()
        labels = data.iloc[:, [17]].values.tolist()

    normalizer = Normalizer()
    objects = normalizer.fit_transform(objects)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    objects, labels = shuffle(objects, labels)

    return objects, labels
