import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=42
)

model = DBSCAN(
    eps=0.2,
    min_samples=5
)

labels = model.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("DBSCAN Clustering")
print("-----------------")
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
