import pandas as pd
import torch


def load_iris():
    data = pd.read_csv('datasets/iris.csv')
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for y, name in class_names.items():
        data.loc[data['species'] == name, 'species'] = y

    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # convert to tensor
    X = torch.from_numpy(X).float()
    std = torch.std(X, 0)
    mean = torch.mean(X, 0)

    X = (X[:] - mean) / std
    Y = torch.from_numpy(Y).long()
    return X, Y, class_names
