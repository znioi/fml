
### 📄 `README.md` – **Experiment 6: Naive Bayes Classifier**

# FML Lab – Experiment 6: Naive Bayes Classifier

## 🎯 Aim

To study and implement the **Naive Bayes Classifier** for binary classification using synthetic data, and visualize the decision regions.

---

## 📚 Theory

### 🌐 What is Naive Bayes?

**Naive Bayes** is a family of **probabilistic classifiers** based on **Bayes' Theorem**, with a strong (naive) assumption that all features are **independent** given the class label.

Despite the simplicity of the assumption, Naive Bayes often performs well in real-world classification tasks such as spam detection, sentiment analysis, and medical diagnosis.

---

### 📖 Bayes' Theorem

Given a class \( C \) and feature vector \( X = (x_1, x_2, ..., x_n) \), Bayes' theorem tells us:

\[
P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}
\]

Since \( P(X) \) is constant for all classes, we only need to maximize:

\[
P(C \mid X) \propto P(X \mid C) \cdot P(C)
\]

---

### 🧮 Naive Assumption

Naive Bayes assumes the **features are conditionally independent**, so:

\[
P(X \mid C) = \prod_{i=1}^{n} P(x_i \mid C)
\]

Hence,

\[
P(C \mid X) \propto P(C) \cdot \prod_{i=1}^{n} P(x_i \mid C)
\]

---

### 📊 Gaussian Naive Bayes

When features are **continuous**, we often assume each feature follows a **Gaussian distribution**:

\[
P(x_i \mid C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} \exp\left(-\frac{(x_i - \mu_C)^2}{2\sigma_C^2}\right)
\]

Where:
- \( \mu_C \): Mean of the feature for class \( C \)
- \( \sigma_C^2 \): Variance of the feature for class \( C \)

---

## 🧪 Code

```python
# Lab 6 - Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# 1. Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_clusters_per_class=1, n_redundant=0, flip_y=0,
                           class_sep=2, random_state=42)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 4. Plot test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors='k')
plt.title('Naive Bayes Decision Scatter')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
````

---

## ✅ Learning Outcomes

* Understood the fundamentals of **Bayes' Theorem** and **probabilistic classification**.
* Learned how to implement **Gaussian Naive Bayes** for real-valued features.
* Explored how the model assumes **feature independence**, which simplifies computation but may not always hold in practice.
* Visualized the decision boundaries created by Naive Bayes on synthetic data.

---

> 💡 *Naive Bayes is a powerful baseline model due to its speed and simplicity, especially for high-dimensional data like text.*

