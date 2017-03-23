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


# def load_qsar():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel
#
#     objects, labels = [None, None]
#     with open('data/qsar.csv', 'r') as f:
#         data = read_csv(f, header=None, delimiter=';')
#         objects = data.iloc[:, range(41)].astype('float64').values.tolist()
#         labels = data.iloc[:, [41]].values.tolist()
#
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_yeast():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel
#
#     objects, labels = [None, None]
#     with open('data/yeast.data', 'r') as f:
#         data = read_csv(f, header=None, delimiter=r"\s+")
#         objects = data.iloc[:, [*range(1, 9)]].astype('float64').values.tolist()
#         labels = data.iloc[:, [9]].values.tolist()
#
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_glass():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel
#
#     objects, labels = [None, None]
#     with open('data/glass.csv', 'r') as f:
#         data = read_csv(f)
#         objects = data.iloc[:, [*range(9)]].astype('float64').values.tolist()
#         labels = data.iloc[:, [9]].values.tolist()
#
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_mushrooms():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/mushrooms.csv', 'r') as f:
#         data = read_csv(f)
#         objects = data.iloc[:, [*range(1, 23)]].values.tolist()
#         labels = data.iloc[:, [0]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_diabetes():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/diabetes.csv', 'r') as f:
#         data = read_csv(f)
#         objects = data.iloc[:, [*range(8)]].values.tolist()
#         labels = data.iloc[:, [-1]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_zoo():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/zoo.csv', 'r') as f:
#         data = read_csv(f)
#         objects = data.iloc[:, [*range(1, 17)]].values.tolist()
#         labels = data.iloc[:, [-1]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_breast_cancer_wisconsin():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/breast_cancer_wisconsin.csv', 'r') as f:
#         data = read_csv(f)
#         objects = data.iloc[:, [*range(2, 32)]].values.tolist()
#         labels = data.iloc[:, [1]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_credit_card():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/credit_card.csv', 'r') as f:
#         data = read_csv(f)
#         objects = data.iloc[:, [*range(1, 24)]].values.tolist()
#         labels = data.iloc[:, [24]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_car():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/car.csv', 'r') as f:
#         data = read_csv(f, header=None)
#         objects = data.iloc[:, [*range(6)]].values.tolist()
#         labels = data.iloc[:, [6]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_abalone():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/abalone.csv', 'r') as f:
#         data = read_csv(f, header=None).dropna()
#         objects = data.iloc[:, [*range(8)]].values.tolist()
#         labels = data.iloc[:, [8]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_transfusion():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/transfusion.csv', 'r') as f:
#         data = read_csv(f).dropna()
#         objects = data.iloc[:, [*range(4)]].values.tolist()
#         labels = data.iloc[:, [4]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_seismic_bumps():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/seismic_bumps.csv', 'r') as f:
#         data = read_csv(f, header=None).dropna()
#         objects = data.iloc[:, [*range(18)]].values.tolist()
#         labels = data.iloc[:, [18]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_king_rook_vs_king_pawn():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/king_rook_vs_king_pawn.csv', 'r') as f:
#         data = read_csv(f, header=None).dropna()
#         objects = data.iloc[:, [*range(36)]].values.tolist()
#         labels = data.iloc[:, [36]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_wine_quality_red():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/winequality_red.csv', 'r') as f:
#         data = read_csv(f, delimiter=";").dropna()
#         objects = data.iloc[:, [*range(11)]].values.tolist()
#         labels = data.iloc[:, [11]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_wine_quality_white():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/winequality_white.csv', 'r') as f:
#         data = read_csv(f, delimiter=";").dropna()
#         objects = data.iloc[:, [*range(11)]].values.tolist()
#         labels = data.iloc[:, [11]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_banknote_authentication():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/banknote_authentication.csv', 'r') as f:
#         data = read_csv(f, header=None).dropna()
#         objects = data.iloc[:, [*range(4)]].values.tolist()
#         labels = data.iloc[:, [4]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_mammographic_masses():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/mammographic_masses.csv', 'r') as f:
#         data = read_csv(f, header=None, na_values=['?']).dropna()
#         objects = data.iloc[:, [*range(1, 5)]].values.tolist()
#         labels = data.iloc[:, [5]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)
#
#
# def load_ilpd():
#     from sklearn.utils import shuffle
#     from sklearn.preprocessing import Normalizer, LabelEncoder
#     from pandas import read_csv
#     from numpy import ravel, transpose, str_
#
#     objects, labels = [None, None]
#     with open('data/ilpd.csv', 'r') as f:
#         data = read_csv(f, header=None, na_values=['?']).dropna()
#         objects = data.iloc[:, [*range(10)]].values.tolist()
#         labels = data.iloc[:, [10]].values.tolist()
#
#     objects = transpose(objects)
#     for i in range(len(objects)):
#         if isinstance(objects[i][0], str_):
#             objects[i] = LabelEncoder().fit_transform(ravel(objects[i]))
#
#     objects = transpose(objects).astype('float64').tolist()
#     normalizer = Normalizer()
#     objects = normalizer.fit_transform(objects)
#     encoder = LabelEncoder()
#     labels = encoder.fit_transform(ravel(labels))
#     objects, labels = shuffle(objects, labels)
#
#     return objects.tolist(), labels.tolist(), len(objects[0]), len(encoder.classes_)