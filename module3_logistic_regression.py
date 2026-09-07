import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
X = np.array([
    [0.5, 1.5],
    [1, 1],
    [1.5, 0.5],
    [3, 0.5],
    [2, 2],
    [1, 2.5]
])

y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Decision Boundary
b = -3
w0 = 1
w1 = 1

x0 = np.linspace(0, 4, 100)
x1 = -(w0 * x0 + b) / w1

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y.ravel())
plt.plot(x0, x1)
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Logistic Regression Decision Boundary")
plt.legend(["Decision Boundary", "Data"])

# SAVE instead of waiting for graph window
plt.savefig("module3_decision_boundary.png", dpi=300, bbox_inches="tight")
plt.close()

# Sigmoid Function
z = np.linspace(-10, 10, 100)
g = sigmoid(z)

plt.figure()
plt.plot(z, g)
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Sigmoid Function")

# SAVE instead of waiting for graph window
plt.savefig("module3_sigmoid.png", dpi=300, bbox_inches="tight")
plt.close()

# Logistic Regression Model
model = LogisticRegression()
model.fit(X, y.ravel())

predictions = model.predict(X)
accuracy = model.score(X, y.ravel())

print("Predictions:", predictions)
print("Accuracy:", accuracy)
print("Coefficients:", model.coef_[0])
print("Intercept:", model.intercept_[0])
print("Graphs saved successfully.")