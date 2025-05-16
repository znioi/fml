Here's a complete `README.md` for **Experiment 4 – Classification using Support Vector Machine (SVM)**, structured for GitHub with readable math expressions and clean markdown formatting:

---

### 📄 `README.md` – **Experiment 4: Support Vector Machine (SVM)**

```markdown
# FML Lab – Experiment 4: Classification using Support Vector Machine (SVM)

## 🎯 Aim

To study and implement **Support Vector Machine (SVM)** for binary classification using a synthetic dataset, and visualize the decision boundary.

---

## 📚 Theory

### What is SVM?

**Support Vector Machine (SVM)** is a powerful supervised learning algorithm used for both **classification** and **regression**. It aims to find the optimal hyperplane that separates data points of different classes **with the maximum margin**.

---

### 🔍 Decision Boundary

For binary classification, SVM tries to solve:

```

maximize:   Margin = 2 / ||w||
subject to: yᵢ (w · xᵢ + b) ≥ 1

```

- `w` → weight vector
- `b` → bias (intercept)
- `xᵢ` → training input vectors
- `yᵢ` → class labels (-1 or +1)

The **decision function** used by SVM is:

```

f(x) = sign(w · x + b)

````

If the data is **not linearly separable**, SVM can use **kernels** (like RBF, polynomial) to project data into higher dimensions where separation is possible.

---

### 💡 Important Concepts

- **Support Vectors**: Data points closest to the hyperplane. They define the margin.
- **Margin**: Distance between the hyperplane and the nearest data point from either class.
- **C (Regularization parameter)**: Controls trade-off between margin width and classification error.
- **Kernel trick**: Allows computation in high-dimensional spaces without explicitly transforming the data.

---

## 🧪 Code

```python
# Lab 4 - Support Vector Machine (SVM) for Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification

# 1. Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train SVM model (linear kernel)
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Plot decision boundary
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Training Data', edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label='Testing Data', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.legend()
plt.show()
````

---

## ✅ Learning Outcomes

* Understood the mathematical foundations of **SVM** and its **decision boundaries**.
* Learned about **linear** and **kernel-based** classification approaches.
* Implemented SVM using `scikit-learn` and visualized the **classification boundary**.
* Interpreted **classification reports** (precision, recall, f1-score).
* Developed insights into **margin maximization**, **support vectors**, and **hyperparameter tuning** (e.g., `C`, `kernel`).

---

> 📌 *SVM is one of the most effective tools for binary classification and provides strong theoretical guarantees with good generalization.*

````

---

### ✅ Optional Additions

- To **show the plot on GitHub**, save the figure as `svm_plot.png` and include:

```markdown
![SVM Decision Boundary](svm_plot.png)
````

* To ensure the repo works on other systems, include a `requirements.txt`:

```txt
numpy
matplotlib
scikit-learn
```

