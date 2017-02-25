#!/usr/bin/env python
import chainer
import chainer.functions as F
import chainer.links as L


# Network definition
# http://data.gunosy.io/entry/2016/07/28/180943
class CNN_1(chainer.Chain):

    def __init__(self, n_out):
        super(CNN_1, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),  # in_channles=1, out_channels=32, ksize=5
            conv2=L.Convolution2D(32, 64, 5), # in_channles=32, out_channels=64, ksize=5
            l1=L.Linear(1024, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)  # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2) # ksize=2
        return self.l1(h)


# Network definition
# http://data.gunosy.io/entry/2016/07/28/180943
class CNN_2(chainer.Chain):

    def __init__(self, n_out):
        super(CNN_2, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),  # in_channels=1, out_channels=32, ksize=5
            conv2=L.Convolution2D(32, 64, 5), # in_channels=32, out_channels=64, ksize=5
            l1=L.Linear(1024, 512),
            l2=L.Linear(512, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)  # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2) # ksize=2
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)


# Network definition
# http://data.gunosy.io/entry/2016/07/28/180943
class CNN_3(chainer.Chain):

    def __init__(self, n_out):
        super(CNN_3, self).__init__(
            conv1=L.Convolution2D(1, 32, 5, pad=1),  # in_channels=1, out_channels=32, ksize=5
            conv2=L.Convolution2D(32, 64, 5, pad=1), # in_channels=32, out_channels=64, ksize=5
            l1=L.Linear(3136, 512),
            l2=L.Linear(512, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, pad=1)  # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, pad=1) # ksize=2
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)


# Network definition
# http://www.iandprogram.net/entry/2016/02/11/181322
class CNN_4(chainer.Chain):

    def __init__(self, n_out):
        super(CNN_4, self).__init__(
            conv1=L.Convolution2D(1, 16, 3),  # in_channels=1, out_channels=16, ksize=3
            conv2=L.Convolution2D(16, 32, 3), # in_channels=16, out_channels=32, ksize=3
            conv3=L.Convolution2D(32, 64, 3), # in_channels=32, out_channels=64, ksize=3
            l1=L.Linear(256, 512),
            l2=L.Linear(512, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)  # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2) # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2) # ksize=2
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)


# Network definition
# http://www.iandprogram.net/entry/2016/02/11/181322
class CNN_5(chainer.Chain):

    def __init__(self, n_out):
        super(CNN_5, self).__init__(
            conv1=L.Convolution2D(1, 16, 3, pad=4),  # in_channels=1, out_channels=16, ksize=3
            conv2=L.Convolution2D(16, 32, 3, pad=2), # in_channels=16, out_channels=32, ksize=3
            conv3=L.Convolution2D(32, 64, 3, pad=1), # in_channels=32, out_channels=64, ksize=3
            conv4=L.Convolution2D(64, 64, 3, pad=1), # in_channels=64, out_channels=64, ksize=3
            l1=L.Linear(576, 256),
            l2=L.Linear(256, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)  # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2) # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2) # ksize=2
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2) # ksize=2
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)
