
### 📄 `README.md` – **Experiment 9: Gaussian Mixture Model (GMM)**

# FML Lab – Experiment 9: Gaussian Mixture Model (GMM)

## 🎯 Aim

To study and implement **Gaussian Mixture Model (GMM)** using the Expectation-Maximization (EM) algorithm for clustering data.

---

## 📚 Theory

### What is Gaussian Mixture Model?

A **Gaussian Mixture Model (GMM)** is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions with unknown parameters.

Unlike K-means which assigns points hard cluster labels, GMM assigns **soft probabilities** of belonging to each cluster, allowing more flexibility.

---

### How does Expectation-Maximization (EM) work in GMM?

1. **Expectation (E-step):** Calculate the probability that each data point belongs to each Gaussian component based on current parameter estimates.

2. **Maximization (M-step):** Update the parameters (means, covariances, and mixing coefficients) of each Gaussian to maximize the likelihood of the data weighted by these probabilities.

3. **Repeat:** Iterate E and M steps until convergence (parameters stabilize).

---

### Model components:

- Each Gaussian \( k \) is defined by mean vector \( \mu_k \) and covariance matrix \( \Sigma_k \).
- Mixing coefficients \( \pi_k \) represent the weight of each Gaussian in the mixture, with \( \sum \pi_k = 1 \).

---

### Advantages

- Captures more complex cluster shapes than K-means.
- Provides soft clustering with probabilities.
- Can model clusters with different shapes and sizes.

---

## 🧪 Code

```python
# Experiment 9 - Gaussian Mixture Model (GMM)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# 1. Generate synthetic dataset
X, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_clusters_per_class=1,
    n_redundant=0,
    flip_y=0,
    class_sep=2,
    random_state=42
)

# 2. Split into train and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# 3. Fit Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2, random_state=42).fit(X_train)

# 4. Predict cluster labels on test data
y_pred = gmm.predict(X_test)

# 5. Evaluate clustering quality using Silhouette Score
print(f"Silhouette Score: {silhouette_score(X_test, y_pred):.2f}")

# 6. Visualize clustered test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, edgecolors='k')
plt.title("GMM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.show()
````

---

## ✅ Learning Outcomes

* Understood the concept of Gaussian Mixture Models and how they differ from K-means.
* Learned the EM algorithm for fitting GMMs.
* Applied GMM clustering on a synthetic dataset.
* Evaluated clustering performance using silhouette score.
* Visualized probabilistic clusters with soft boundaries.

---

> 💡 *GMM provides a flexible clustering approach that models data as a combination of multiple Gaussian distributions.*


