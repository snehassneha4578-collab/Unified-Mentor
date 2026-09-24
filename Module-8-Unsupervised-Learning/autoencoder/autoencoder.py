import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(42)
tf.random.set_seed(42)

X_train = np.random.normal(
    loc=0,
    scale=1,
    size=(1000, 10)
)

autoencoder = models.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(6, activation="relu"),
    layers.Dense(3, activation="relu"),
    layers.Dense(6, activation="relu"),
    layers.Dense(10, activation="linear")
])

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

X_test_normal = np.random.normal(
    loc=0,
    scale=1,
    size=(100, 10)
)

X_test_anomaly = np.random.normal(
    loc=5,
    scale=1,
    size=(20, 10)
)

X_test = np.vstack([
    X_test_normal,
    X_test_anomaly
])

X_reconstructed = autoencoder.predict(
    X_test,
    verbose=0
)

reconstruction_error = np.mean(
    np.square(X_test - X_reconstructed),
    axis=1
)

threshold = np.percentile(
    reconstruction_error[:100],
    95
)

anomalies = reconstruction_error > threshold

print("Autoencoder Anomaly Detection")
print("-----------------------------")
print("Reconstruction Error Threshold:", threshold)
print("Detected Anomalies:", np.sum(anomalies))

plt.figure(figsize=(10, 5))

plt.plot(
    reconstruction_error,
    marker="o",
    linestyle="",
    markersize=4
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.title("Autoencoder Anomaly Detection")
plt.xlabel("Sample")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()
