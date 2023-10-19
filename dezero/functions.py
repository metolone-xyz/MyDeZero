import numpy as np
from dezero.core import Function
from dezero.core import as_variable


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


# 行列の形状変更
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape  # 変形する形状をshapeで受け取る。

    def forward(self, x):
        self.x_shape = x.shape
        # reshape関数・・・Numpyの関数で形状変更
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)  # Variableインスタンスへ変換
    return Reshape(shape)(x)

# 転置行列
class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        gx = np.transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)