import numpy as np
import unittest
import weakref
import contextlib

#複雑な計算グラフ

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None #微分した値
        self.creator = None #関数の出力結果
        self.generation = 0 #世代を記録（処理の優先度に対応）

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    #retain_gradがTrueのときすべての変数が勾配を保存
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #dy/dyの自動追加

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop() #関数を取得
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)#複数の関数の入出力値を取得
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

def as_array(x):
    if np.isscalar(x): #スカラ系の型を判定
        return np.array(x)
    return x

class Function:
    #アスタリスク・・・可変長引数
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] #リスト内包表記
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs] #弱参照を作り循環をなくす
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, xs):
        raise NotImplementedError() #まだ実装されていない部分を表す
    #逆伝播
    def backward(self, gys):
        raise NotImplementedError()

#インスタンス化しない
class Config:
    enable_backprop = True
#継承
class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy #gyは出力側から伝わる微分が渡される。
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

#中心差分近似を用いて数値微分を求める関数を実装する
def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
def no_grad():
    return using_config('enable_backprop', False)

with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)