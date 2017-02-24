#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np


# Network definition
# http://data.gunosy.io/entry/2016/07/28/180943
class CNN(chainer.Chain):

    def __init__(self, n_out):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),  # in_channles=1, out_channels=32, ksize=5
            conv2=L.Convolution2D(32, 64, 5), # in_channles=32, out_channels=64, ksize=5
            l1=L.Linear(1024, n_out)
        )

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2)  # ksize=2
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2) # ksize=2
        return self.l1(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result_cnn_v1',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noise', '-n', action='store_true', default=False,
                        help='Use MNIST with noises')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('MNIST noise: {}'.format(args.noise))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(CNN(10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    _train, _test = chainer.datasets.get_mnist(ndim=3)

    # Add noise
    if not args.noise:
        train, test = _train, _test
    else:
        args.out += '_noise'
        train_data, train_label = [], []
        test_data, test_label = [], []
        np.random.seed(12345)
        for k in range(4):
            for train_orig in _train:
                noised = np.random.normal(0, 0.05, (len(train_orig[0][0]), len(train_orig[0][0][0])))
                for j, row in enumerate(train_orig[0][0]):
                    for i, d in enumerate(row):
                        noised[j][i] += d
                        if noised[j][i] < 0:
                            noised[j][i] = 0.0
                        elif 1 < noised[j][i]:
                            noised[j][i] = 1.0
                train_data.append(np.array([noised], dtype=np.float32))
                train_label.append(np.int32(train_orig[1]))
            for test_orig in _test:
                noised = np.random.normal(0, 0.05, (len(test_orig[0][0]), len(test_orig[0][0][0])))
                for j, row in enumerate(test_orig[0][0]):
                    for i, d in enumerate(row):
                        noised[j][i] += d
                        if noised[j][i] < 0:
                            noised[j][i] = 0.0
                        elif 1 < noised[j][i]:
                            noised[j][i] = 1.0
                test_data.append(np.array([noised], dtype=np.float32))
                test_label.append(np.int32(train_orig[1]))
            train = chainer.datasets.tuple_dataset.TupleDataset(train_data, train_label)
            test = chainer.datasets.tuple_dataset.TupleDataset(test_data, test_label)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',
                              file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                              'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
