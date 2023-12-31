import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    #逆伝搬の際には順伝搬で用いたデータを使用し、合成関数の導関数を求める

#微分の計算を行う逆伝播の機能とforwardメソッドを呼ぶ際に、入力されたVariableインスタンスを保存する機能
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input #入力された変数を覚える。
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

#合成関数の微分
A = Square()
B = Exp()
C = Square()

#順伝搬
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#逆伝播
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)