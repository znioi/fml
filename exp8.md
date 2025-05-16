
### 📄 `README.md` – **Experiment 8: K-means Clustering**

# FML Lab – Experiment 8: K-means Clustering

## 🎯 Aim

To study and implement **K-means Clustering** for unsupervised learning to find natural groupings (clusters) in the Wine dataset.

---

## 📚 Theory

### What is K-means Clustering?

K-means is a popular **unsupervised machine learning algorithm** used for clustering. It partitions data into **K distinct clusters** based on feature similarity.

---

### How does K-means work?

1. **Initialize**: Randomly choose K points as cluster centroids.
2. **Assign**: Assign each data point to the nearest centroid based on distance (usually Euclidean).
3. **Update**: Recalculate centroids as the mean of all points assigned to that cluster.
4. **Repeat**: Repeat assignment and update steps until centroids stabilize or maximum iterations are reached.

---

### Objective function

K-means tries to minimize the **within-cluster sum of squares (inertia)**:

```

J = Σ (over k clusters) Σ (over points in cluster) || x\_i - μ\_k ||^2

````

where:
- \( x_i \) = data points
- \( μ_k \) = centroid of cluster k

Lower inertia means tighter clusters.

---

## 🧪 Code

```python
# Experiment 8 - K-means Clustering

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Wine dataset
wine = datasets.load_wine()

# 2. Standardize features for better clustering
X_scaled = StandardScaler().fit_transform(wine.data)

# 3. Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)

# 4. Get cluster labels and centroids
labels, centroids = kmeans.labels_, kmeans.cluster_centers_

# 5. Visualize clusters and centroids (using first two features)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)

plt.title('K-means Clustering of Wine Dataset')
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.colorbar(scatter, label='Cluster')
plt.show()

# 6. Print cluster sizes and inertia
print(f"Samples per cluster: {np.bincount(labels)}")
print(f"Inertia: {kmeans.inertia_:.2f}")
````

---

## ✅ Learning Outcomes

* Understood the concept of unsupervised learning and clustering.
* Learned how K-means iteratively assigns clusters and updates centroids.
* Applied K-means on a real dataset after feature scaling.
* Visualized clusters and centroids in 2D space.
* Evaluated clustering quality using inertia and cluster size distribution.

---

> 💡 *K-means requires specifying the number of clusters beforehand and is sensitive to initial centroid placement and feature scaling.*
