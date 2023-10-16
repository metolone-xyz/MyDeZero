import numpy as np
import unittest

#逆伝播の自動化(再帰)

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None #微分した値
        self.creator = None #関数の出力結果

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #dy/dyの自動追加
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #関数を取得
            x, y = f.input, f.output    #関数の入出力値を取得
            x.grad = f.backward(y.grad) #backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)

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
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs #出力も覚える
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, xs):
        raise NotImplementedError() #まだ実装されていない部分を表す
    #逆伝播
    def backward(self, gys):
        raise NotImplementedError()

#継承
class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.input.data
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


x0 = Variable(np.array(2))
x1 = Variable(np.array(2))
y = add(x0, x1)
print(y.data)