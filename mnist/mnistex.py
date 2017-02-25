#!/usr/bin/env python
import chainer
import numpy as np


def gen_mnist_with_noise(train_raw, test_raw, repeat=4, seed=12345, scale=0.02):
    train_data, train_label = [], []
    test_data, test_label = [], []
    np.random.seed(seed)
    for k in range(repeat):
        for raw in train_raw:
            noised = np.random.normal(0, scale, (28, 28))
            for j in range(28):
                for i in range(28):
                    noised[j][i] += raw[0][0][j][i]
                    if noised[j][i] < 0:
                        noised[j][i] = 0.0
                    elif 1 < noised[j][i]:
                        noised[j][i] = 1.0
            train_data.append(np.array([noised], dtype=np.float32))
            train_label.append(np.int32(raw[1]))
        for raw in test_raw:
            noised = np.random.normal(0, scale, (28, 28))
            for j in range(28):
                for i in range(28):
                    noised[j][i] += raw[0][0][j][i]
                    if noised[j][i] < 0:
                        noised[j][i] = 0.0
                    elif 1 < noised[j][i]:
                        noised[j][i] = 1.0
            test_data.append(np.array([noised], dtype=np.float32))
            test_label.append(np.int32(raw[1]))
    train = chainer.datasets.tuple_dataset.TupleDataset(train_data, train_label)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_data, test_label)
    return train, test

def gen_mnist_with_shift(train_raw, test_raw):
    train_data, train_label = [], []
    test_data, test_label = [], []
    shifts = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (0, -2), (0, 2), (-2, 0), (2, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for k in shifts:
        for raw in train_raw:
            shifted = [[ 0.0 for __ in range(28) ] for _ in range(28) ]
            for j in range(28):
                for i in range(28):
                    try:
                        shifted[j][i] = raw[0][0][j + k[0]][i + k[1]]
                    except IndexError:
                        shifted[j][i] = 0.0
            train_data.append(np.array([shifted], dtype=np.float32))
            train_label.append(np.int32(raw[1]))
        for raw in test_raw:
            shifted = [[ 0.0 for __ in range(28) ] for _ in range(28) ]
            for j in range(28):
                for i in range(28):
                    try:
                        shifted[j][i] = raw[0][0][j + k[0]][i + k[1]]
                    except IndexError:
                        shifted[j][i] = 0.0
            test_data.append(np.array([shifted], dtype=np.float32))
            test_label.append(np.int32(raw[1]))
    train = chainer.datasets.tuple_dataset.TupleDataset(train_data, train_label)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_data, test_label)
    return train, test
