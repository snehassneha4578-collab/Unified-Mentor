# Module 8 — Unsupervised Learning

This repository contains my **Module 8 Internship Assignment** focused on unsupervised learning techniques, anomaly detection, and autoencoders.

## Topics Covered

* Clustering
* K-Means Clustering
* DBSCAN
* Bisecting K-Means
* Outlier Detection
* Isolation Forest
* Local Outlier Factor (LOF)
* Autoencoders
* Anomaly Detection using Reconstruction Error

## 1. Clustering

Clustering is an unsupervised machine learning technique used to group similar data points together without predefined labels.

### Algorithms Covered

* K-Means
* DBSCAN
* Bisecting K-Means

## 2. Outlier Detection

Outlier detection identifies observations that significantly differ from the majority of the data.

### Techniques Covered

* Isolation Forest
* Local Outlier Factor (LOF)

## 3. Autoencoders

An autoencoder is a neural network that learns to reconstruct its input.

It consists of:

```text
Input → Encoder → Latent Representation → Decoder → Reconstruction
```

The reconstruction error can be used for anomaly detection. Data points with unusually high reconstruction error may be considered anomalous.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* TensorFlow
* Keras

## Repository Structure

```text
Module-8-Unsupervised-Learning/
│
├── README.md
├── clustering/
│   ├── kmeans.py
│   ├── dbscan.py
│   └── bisecting_kmeans.py
│
├── outlier_detection/
│   ├── isolation_forest.py
│   └── local_outlier_factor.py
│
├── autoencoder/
│   └── autoencoder.py
│
├── requirements.txt
└── references.md
```

## Learning Outcomes

Through this module, I learned how to:

* Perform unsupervised clustering.
* Group data using K-Means.
* Detect density-based clusters using DBSCAN.
* Apply Bisecting K-Means.
* Identify anomalous observations.
* Use Isolation Forest for anomaly detection.
* Apply Local Outlier Factor.
* Understand the architecture of autoencoders.
* Use reconstruction error for anomaly detection.

## References

The implementations and concepts in this assignment were studied using the official Scikit-learn and TensorFlow documentation listed in `references.md`.

## Author

**Sneha**

ECE Student | AI/ML | Embedded Systems | VLSI
