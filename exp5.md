

### 📄 `README.md` – **Experiment 5: Bagging using Random Forests**

# FML Lab – Experiment 5: Bagging using Random Forests

## 🎯 Aim

To study and implement **Bagging (Bootstrap Aggregating)** technique using **Random Forests** for classification, and evaluate the model using a confusion matrix.

---

## 📚 Theory

### 🌳 What is Bagging?

**Bagging**, or **Bootstrap Aggregating**, is an ensemble learning technique that improves the stability and accuracy of machine learning algorithms. It reduces variance and helps avoid overfitting.

Steps:
1. Generate multiple datasets using **bootstrapping** (sampling with replacement).
2. Train a **base learner** (e.g., decision tree) on each dataset.
3. Aggregate predictions (for classification: **majority voting**).

### 🌲 What is a Random Forest?

A **Random Forest** is an ensemble of decision trees built using the **bagging** approach with an additional layer of randomness.

**Key Features:**
- Each tree is trained on a bootstrapped sample.
- At each split, only a random subset of features is considered.
- Final prediction is made by **majority voting** among all trees.

---

### 📐 Mathematical Insight

If we denote:

- \( h_1(x), h_2(x), \dots, h_n(x) \) as predictions from \( n \) individual trees,
- Then the final prediction \( H(x) \) is given by:

\[
H(x) = \text{mode}\{ h_1(x), h_2(x), \dots, h_n(x) \}
\]

This majority vote helps improve **accuracy** and **generalization**.

---

## 🧪 Code

```python
# Lab 5 - Bagging with Random Forest Classifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# 1. Generate synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 2. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train Random Forest classifier (with 100 trees)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Predict and evaluate using confusion matrix
y_pred = rf_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# 5. Visualize the confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
````

---

## ✅ Learning Outcomes

* Understood the **Bagging** approach and its importance in reducing model variance.
* Implemented **Random Forest**, an ensemble of decision trees using `scikit-learn`.
* Visualized the **confusion matrix** to evaluate classification performance.
* Observed how ensemble models outperform single decision trees in most cases.

---

> 💡 *Bagging, especially with Random Forests, is a powerful technique for building accurate and stable classifiers, especially in high-dimensional datasets.*


