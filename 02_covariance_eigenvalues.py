import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [2, 4, 5, 4, 6]})
print("Covariance Matrix:\n", df.cov())

cov = df.cov().values
values, vectors = np.linalg.eig(cov)
print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

plt.scatter(df["X"], df["Y"])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Covariance Data")
plt.show()
