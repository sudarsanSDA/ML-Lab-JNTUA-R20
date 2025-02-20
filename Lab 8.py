import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Applying the Expectation-Maximization (EM) algorithm with Gaussian Mixture Model
em_model = GaussianMixture(n_components=3, random_state=42)
em_model.fit(X)
em_labels = em_model.predict(X)

# Applying k-Means Clustering
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X)

# Evaluate Clustering using Adjusted Rand Score
em_ari = adjusted_rand_score(y, em_labels)
kmeans_ari = adjusted_rand_score(y, kmeans_labels)

# Printing the comparison results
print(f"Adjusted Rand Index for EM (Gaussian Mixture): {em_ari:.4f}")
print(f"Adjusted Rand Index for K-Means: {kmeans_ari:.4f}")

# Visualizing the clusters
plt.figure(figsize=(12, 6))

# EM Clustering Visualization
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='viridis')
plt.title("EM Algorithm (Gaussian Mixture) Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# K-Means Clustering Visualization
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.show()
