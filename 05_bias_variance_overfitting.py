import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(1)
X = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.15, 20)

X_test = np.linspace(0, 1, 200)
plt.scatter(X, y, label="Data")

for degree in [1, 4, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X.reshape(-1, 1), y)
    plt.plot(X_test, model.predict(X_test.reshape(-1, 1)), label=f"Degree {degree}")

plt.legend()
plt.title("Underfitting, Good Fit and Overfitting")
plt.show()
