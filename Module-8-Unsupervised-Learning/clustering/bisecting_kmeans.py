import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import BisectingKMeans

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.0,
    random_state=42
)

model = BisectingKMeans(
    n_clusters=4,
    random_state=42
)

model.fit(X)

labels = model.labels_
centers = model.cluster_centers_

print("Bisecting K-Means")
print("-----------------")
print("Cluster Centers:")
print(centers)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("Bisecting K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
