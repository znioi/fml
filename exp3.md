
---

### 📄 `README.md` (for Lab 3 – KNN)

```markdown
# FML Lab – Experiment 3: K-Nearest Neighbors (KNN)

## 🎯 Aim

To study and implement the **K-Nearest Neighbors (KNN)** algorithm using a synthetic classification dataset, and visualize the decision boundary.

---

## 📚 Theory

### What is KNN?

**K-Nearest Neighbors (KNN)** is a simple, intuitive, and powerful **non-parametric**, **instance-based** supervised learning algorithm used for classification and regression. It works on the principle that similar data points are close to each other in feature space.

---

### 🔍 Working Principle

Given a new data point:

1. Calculate the **distance** from the query point to all points in the training set.
2. Select the **K nearest neighbors** (using a distance metric like Euclidean distance).
3. Take a **majority vote** among the neighbors' classes for classification.

---

### 📐 Euclidean Distance

The most commonly used distance metric is the **Euclidean distance**, defined as:

```

distance(x, y) = sqrt((x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)²)

````

This formula computes the straight-line distance between two points in **n-dimensional space**.

---

### ⚙️ Properties

- **Lazy learner**: No model is explicitly trained during the fit phase.
- **No assumptions** about data distribution.
- Sensitive to **feature scaling** and **outliers**.
- Requires careful selection of **K** for optimal performance:
  - Small K → more sensitive to noise (overfitting)
  - Large K → may smooth over important patterns (underfitting)

---

## 🧪 Code

```python
# Lab 3 - K Nearest Neighbors (KNN)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Generate synthetic dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Create a mesh grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict labels for the grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(cmap_bold),
            edgecolor='k', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(cmap_bold),
            marker='x', edgecolor='k', label='Test')
plt.legend()
plt.title(f'K-Nearest Neighbors (k={k}) Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
````

---

## ✅ Learning Outcomes

* Understood the concept and **working of the KNN algorithm**.
* Implemented KNN using `scikit-learn` with synthetic data.
* Visualized how **decision boundaries change** with different values of K.
* Learned about the trade-offs between **model complexity and performance**.
* Explored how **instance-based learning** works in contrast to parametric methods like logistic regression.

---

> 📌 *KNN is a powerful baseline model and an excellent introduction to distance-based classification techniques.*

````

---

### 💡 Tips for GitHub

GitHub does **not render LaTeX math expressions** inline like Jupyter or Overleaf. So:
- Use code blocks (```) for all equations.
- Use markdown formatting for structure.
- Optionally, add an image of the KNN decision boundary in your repo and reference it like:

```markdown
![KNN Decision Boundary](knn_plot.png)
````

