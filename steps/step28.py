import numpy as np
import math
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph

def rosenbrock(x0, x1):
    y = 100*(x1 - x0**2)**2 + (x0 - 1)**2
    return y

#___________________実装____________________
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 #学習率
iters = 10000 #繰り返す回数

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr*x0.grad
    x1.data -= lr*x1.grad


