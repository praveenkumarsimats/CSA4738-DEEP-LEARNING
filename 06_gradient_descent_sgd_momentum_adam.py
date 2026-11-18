import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def gradient(x):
    return 2*x

def optimize(method, lr=0.1, epochs=30):
    x = 5.0
    v = 0.0
    m = 0.0
    s = 0.0
    values = []
    for t in range(1, epochs + 1):
        g = gradient(x)
        if method == "GD":
            x -= lr * g
        elif method == "SGD":
            x -= lr * g + np.random.normal(0, 0.02)
        elif method == "Momentum":
            v = 0.9 * v + lr * g
            x -= v
        else:
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g**2
            mh = m / (1 - 0.9**t)
            sh = s / (1 - 0.999**t)
            x -= lr * mh / (np.sqrt(sh) + 1e-8)
        values.append(f(x))
    return values

for method in ["GD", "SGD", "Momentum", "Adam"]:
    plt.plot(optimize(method), label=method)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.show()
