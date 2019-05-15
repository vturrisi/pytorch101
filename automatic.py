import sys
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_csv_dataset import load_iris

torch.manual_seed(1)
np.random.seed(1)


class MLP(nn.Module):
    def __init__(self, arch):
        super().__init__()

        self.layers = nn.Sequential()
        for i, (inp, out) in enumerate(zip(arch[:-2], arch[1:-1])):
            self.layers.add_module('linear-{}'.format(i + 1), nn.Linear(inp, out))
            self.layers.add_module('relus-{}'.format(i + 1), nn.ReLU())
        self.layers.add_module('linear-{}'.format(i + 2), nn.Linear(arch[-2], arch[-1]))

    def forward(self, X):
        o = self.layers(X)
        return o

if __name__ == '__main__':
    # prepare csv data
    X, Y, class_names = load_iris()

    loss_function = nn.CrossEntropyLoss()
    mlp_auto = MLP([X.size(1), 20, 50, 20, len(class_names)])
    print(mlp_auto)
    optimiser = optim.SGD(mlp_auto.parameters(), lr=0.001, momentum=0.9)

    mlp_auto.train()
    for epoch in range(10):
        # shuffle data
        index = torch.randperm(X.size(0))
        X = X[index]
        Y = Y[index]

        rights = 0
        total = X.size(0)
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(zip(X, Y)):

                x = x.view(1, -1)
                y = y.view(-1)

                yhat = mlp_auto(x)
                loss = loss_function(yhat, y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                # print(mlp_auto.layers[-1].weight)

                v, prediction = torch.max(yhat, 1)
                prediction = prediction.item()
                if prediction == y.item():
                    rights += 1

        print(rights / total)
