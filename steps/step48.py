import dezero
import matplotlib.pyplot as plt

x, t = dezero.datasets.get_spiral(train=True)


# matplotlibを使用してデータをプロット
plt.figure(figsize=(10, 8))
for i in range(3):  # 3クラス
    plt.scatter(x[t == i][:, 0], x[t == i][:, 1], label=f"Class {i}")
plt.title("Spiral Data Visualization")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
