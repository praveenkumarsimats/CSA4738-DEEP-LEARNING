import numpy as np
import matplotlib.pyplot as plt

v = np.array([1, 2, 3])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Vector:", v)
print("Matrix A:\n", A)
print("A + B:\n", A + B)
print("A x B:\n", A @ B)
print("Transpose A:\n", A.T)
print("Vector norm:", np.linalg.norm(v))

plt.bar(["1", "2", "3"], v)
plt.title("Vector Values")
plt.show()
