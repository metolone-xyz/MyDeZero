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
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self) #出力変数に生みの親を覚えさせる。
        self.input = input
        self.output = output #出力も覚える
        return output
    def forward(self, x):
        raise NotImplementedError() #まだ実装されていない部分を表す
    #逆伝播
    def backward(self, gy):
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


#Square
#合成関数の微分
A = Square()
B = Exp()
C = Square()

#順伝搬
x = Variable(np.array(0.5))
y = square(exp(square(x)))  #関数化するとまとめて書くこともできる！

#逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)