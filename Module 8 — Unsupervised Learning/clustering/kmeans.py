import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X)

labels = model.labels_
centers = model.cluster_centers_

print("K-Means Clustering")
print("-------------------")
print("Cluster Centers:")
print(centers)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
