import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

x = np.arange(0, 11)
p = binom.pmf(x, 10, 0.5)
plt.stem(x, p)
plt.title("Binomial Distribution")
plt.show()

prior = 0.01
sensitivity = 0.95
false_positive = 0.05
posterior = (sensitivity * prior) / (sensitivity * prior + false_positive * (1 - prior))
print("P(Disease | Positive) =", posterior)
