# Kakimono Friends

## Installation

Python 2.7.13

```
pip install chainer=1.21.0
pip install matplotlib=2.0.0
```

If necessary,
```
export CHAINER_DATASET_ROOT="/path/to/chainer/dataset"
```

## Execution

For example,
```
cd mnist
./train_mnist_cnn_v1.py
dot -Tpng result_cnn_v1/cg.dot -o result_cnn_v1/cg.png
```
