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
