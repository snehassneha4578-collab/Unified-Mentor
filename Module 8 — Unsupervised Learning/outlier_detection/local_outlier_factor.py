import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.RandomState(42)

normal_data = 0.3 * rng.randn(300, 2)

outliers = rng.uniform(
    low=-4,
    high=4,
    size=(20, 2)
)

X = np.vstack([
    normal_data,
    outliers
])

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.06
)

predictions = model.fit_predict(X)

normal = X[predictions == 1]
anomalies = X[predictions == -1]

print("Local Outlier Factor")
print("--------------------")
print("Total data points:", len(X))
print("Detected outliers:", len(anomalies))

plt.scatter(
    normal[:, 0],
    normal[:, 1],
    label="Normal"
)

plt.scatter(
    anomalies[:, 0],
    anomalies[:, 1],
    marker="x",
    s=100,
    label="Outlier"
)

plt.title("Local Outlier Factor")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
