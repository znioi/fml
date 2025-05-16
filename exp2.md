# FML Lab – Experiment 2: Logistic Regression

## 🎯 Aim
To study and implement **Logistic Regression** using the Iris dataset, perform binary classification, and visualize the decision boundary.

---

## 📚 Theory

**Logistic Regression** is a fundamental supervised learning algorithm used for **binary classification**. It predicts the probability of a class label based on a linear combination of input features passed through a **sigmoid function**.

The sigmoid function is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where } z = w^T x + b
\]

- \( x \) is the input feature vector
- \( w \) are the learned weights
- \( b \) is the bias term

### 🔍 Classification Rule

After computing the probability \( \hat{y} = \sigma(z) \), we apply a threshold:
- \( \hat{y} \geq 0.5 \Rightarrow \) Predict class 1
- \( \hat{y} < 0.5 \Rightarrow \) Predict class 0

### 🧮 Loss Function

To train the model, we minimize the **cross-entropy loss**:

\[
\mathcal{L} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
\]

This function penalizes confident but wrong predictions more heavily.

### 📊 Application to Iris Dataset

- The Iris dataset has 3 flower classes. We convert it into a binary classification problem: "Is the flower *Iris-virginica* or not?"
- Only the **first two features** are used to enable easy 2D visualization of the decision boundary.

---

## 🧪 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# Load dataset
data = load_iris()
X = data.data[:, :2]  # Use only first two features for 2D visualization
y = (data.target == 2).astype(int)  # Binary classification: Is class 2 or not

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Decision Boundary Visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'blue']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['darkred', 'darkblue']))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary of Logistic Regression")
plt.show()
