import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

#注意点
#Functionクラスで実装するメソッドは、Variableインスタンスを入力とし、Variableインスタンスを出力
#Variableインスタンスの実際のデータはインスタンス変数のdataに存在すること

#基底クラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError() #まだ実装されていない部分を表す

#Functionクラスを継承したSquareクラス
class Square(Function):
    def forward(self, x):
        return x**2

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)   #100