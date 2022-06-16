import math

import numpy as np
from ..metric.metric import *
from ..conversion.conversion import *
from abc import ABCMeta
from abc import abstractmethod
from numpy import ndarray
from ..optimizer.optimizer import BaseOptimizer


class BaseLayer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def backward(self, dy: ndarray) -> ndarray:
        pass

    @abstractmethod
    def init(self, input_shape: tuple) -> tuple:
        pass


class WeightLayer(metaclass=ABCMeta):
    @abstractmethod
    def update_grad(self):
        pass


class LastLayer(metaclass=ABCMeta):
    @abstractmethod
    def set_t(self, t):
        pass

    @abstractmethod
    def predict(self):
        pass


class Relu(BaseLayer):
    def init(self, input_shape: tuple) -> tuple:
        return input_shape

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return relu(x)

    def backward(self, dy):
        return dy * relu_grad(self.x)


class Sigmoid(BaseLayer):
    def init(self, input_shape: tuple) -> tuple:
        return input_shape

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, dy):
        return dy * sigmoid_grad(self.x)


class Affine(BaseLayer, WeightLayer):
    def init(self, input_shape: tuple) -> tuple:
        in_node_len = 1
        for size in input_shape:
            in_node_len *= size
        init_scale = math.sqrt(2.0 / in_node_len)
        self.w = init_scale * np.random.rand(in_node_len, self.out_nodes_len)
        self.b = np.zeros(self.out_nodes_len)
        return self.out_nodes_len,

    def __init__(self, out_nodes_len: int, optimizer):
        self.out_nodes_len = out_nodes_len
        self.w = None
        self.b = None

        self.x = None

        self.dw = None
        self.db = None
        self.optimizer: BaseOptimizer = optimizer

    def update_grad(self):
        self.optimizer.update([self.w, self.b], [self.dw, self.db])

    def forward(self, x):
        self.x = x
        x = x.reshape(x.shape[0], -1)

        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dy):
        original_x_shape = self.x.shape
        x = self.x.reshape(self.x.shape[0], -1)

        dx = np.dot(dy, self.w.T)
        self.dw = np.dot(x.T, dy)
        self.db = np.sum(dy, axis=0)

        dx = dx.reshape(*original_x_shape)
        return dx


class SoftmaxWithLoss(BaseLayer, LastLayer):
    def predict(self, x):
        self.y = softmax(x)
        return self.y.argmax(axis=1)

    def init(self, input_shape: tuple) -> tuple:
        return input_shape

    def set_t(self, t):
        self.t = t

    def __init__(self):
        self.t = None
        self.loss = None
        self.y: ndarray = None

    def forward(self, x):
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dy):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout(BaseLayer):
    """
    http://arxiv.org/abs/1207.0580
    """

    def init(self, input_shape: tuple) -> tuple:
        return input_shape

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dy):
        return dy * self.mask


class Convolution(BaseLayer, WeightLayer):
    def init(self, input_shape: tuple) -> tuple:
        channel_len, in_size, in_size = input_shape
        init_scale = math.sqrt(2.0 / (channel_len * self.filter_size * self.filter_size))
        self.w = init_scale * np.random.rand(
            self.filter_len, channel_len, self.filter_size, self.filter_size
        )
        self.b = np.zeros(self.filter_len)
        if self.pad is None:
            self.out_size = in_size
            self.pad = (self.stride * in_size - self.stride - in_size + self.filter_size) // 2
        else:
            self.out_size = 1 + (in_size + 2 * self.pad - self.filter_size) // self.stride
        return self.filter_len, self.out_size, self.out_size

    def update_grad(self):
        self.optimizer.update([self.w, self.b], [self.dw, self.db])

    def __init__(
            self, filter_len, filter_size, stride, pad,
            optimizer: BaseOptimizer
    ):
        self.filter_len = filter_len
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.optimizer = optimizer

        self.x = None
        self.col = None
        self.col_W = None

        self.out_size = 0
        self.w = None
        self.b = None

        self.dw = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.w.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, self.out_size, self.out_size, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dy):
        FN, C, FH, FW = self.w.shape
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dy, axis=0)
        self.dw = np.dot(self.col.T, dy)
        self.dw = self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dy, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling(BaseLayer):
    def init(self, input_shape: tuple) -> tuple:
        channel_len, in_size, in_size = input_shape
        self.out_size = 1 + (in_size - self.pool_size) // self.stride
        return channel_len, self.out_size, self.out_size

    def __init__(self, pool_size, stride, pad):
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad

        self.out_size = 0

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_size) / self.stride)
        out_w = int(1 + (W - self.pool_size) / self.stride)
        col = im2col(x, self.pool_size, self.pool_size, self.stride, self.pad)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dy):
        dy = dy.transpose(0, 2, 3, 1)

        pool_size = self.pool_size * self.pool_size
        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(
            self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_size, self.pool_size, self.stride,
                    self.pad)

        return dx
