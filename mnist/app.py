#!/usr/bin/env python
from flask import Flask, request
app = Flask(__name__)

import os
import test_mnist_cnn

@app.route('/', methods=['POST'])
def hello():
    return str(test_mnist_cnn.test_mnist(
        1,
        '{}/result_cnn1/model.npz'.format(os.path.abspath(os.path.dirname(__file__))),
        request.form['inputdata']
    )) + '\n'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
