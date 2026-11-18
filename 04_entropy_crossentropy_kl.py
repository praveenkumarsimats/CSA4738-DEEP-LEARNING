import numpy as np
from scipy.stats import entropy

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.6, 0.3, 0.1])

H = -np.sum(p * np.log2(p))
CE = -np.sum(p * np.log2(q))
KL = entropy(p, q, base=2)

print("Entropy:", H)
print("Cross-Entropy:", CE)
print("KL Divergence:", KL)
