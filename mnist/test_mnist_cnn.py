#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
import cnn


def test_mnist(_cnn, _model, _inputdata):
    if _cnn == 1:
        cnnobj = cnn.CNN_1(10)
    elif _cnn == 2:
        cnnobj = cnn.CNN_2(10)
    elif _cnn == 3:
        cnnobj = cnn.CNN_3(10)
    elif _cnn == 4:
        cnnobj = cnn.CNN_4(10)
    elif _cnn == 5:
        cnnobj = cnn.CNN_5(10)
    model = L.Classifier(cnnobj)

    # Load the model
    chainer.serializers.load_npz(_model, model)

    # Evaluate
    input1d = [ 1.0 - float(d) / 255.0 for d in _inputdata.split(',') ]
    input2d = [[ input1d[j * 28 + i] for i in range(28) ] for j in range(28) ]
    input4d = np.array([[input2d]], dtype=np.float32)
    evaluated = cnnobj(input4d)
    result = np.argmax(evaluated.data[0])
    print(result)
    return result

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--cnn', '-c', type=int, default=1,
                        help='CNN type (1--5)')
    parser.add_argument('--model', '-m', default='result/model.npz',
                        help='Path to serialized model')
    parser.add_argument('--inputdata', '-i', default=','.join([ '0' for _ in range(28 * 28) ]),
                        help='Input data with comma separators (0:white--255:black)')
    args = parser.parse_args()

    #print('CNN: cnn{}'.format(args.cnn))
    #print('Model: {}'.format(args.model))
    #print('Input: {}'.format(args.inputdata))
    #print('')

    test_mnist(args.cnn, args.model, args.inputdata)

if __name__ == '__main__':
    main()
