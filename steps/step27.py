import numpy as np
import math
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

#テイラー展開の実装・・・sin関数をテイラー展開を元に実装
def my_sin(x, threshold= 1e-150): #threashold・・・閾値
    y = 0
    for i in range(100000):
        c = (-1)**i / math.factorial(2*i + 1) #階上計算
        t = c * x**(2*i + 1)
        y = y + t
        # 閾値を下回ったらforを抜ける
        if abs(t.data) < threshold:
            break
    return y

#___________________実装____________________
x = Variable(np.array(np.pi/4))
y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='mysin.png')