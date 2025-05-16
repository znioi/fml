
---


# FML Lab - Experiment 2

## Aim

To study and implement **Logistic Regression** using the Iris dataset and visualize the decision boundary for binary classification.

---

## Theory

Logistic Regression is a **supervised learning algorithm** used for classification problems. Unlike Linear Regression which outputs continuous values, Logistic Regression predicts discrete classes—often 0 or 1 in binary classification.

It uses the **sigmoid function** to map the linear combination of inputs to a value between 0 and 1, which can be interpreted as a probability:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:

- \( z = w^T x + b \)
- \( w \) is the weight vector
- \( x \) is the input feature vector
- \( b \) is the bias term

The algorithm optimizes the parameters \( w \) and \( b \) by minimizing the **log-loss (cross-entropy loss)** using gradient descent or other solvers.

In this experiment, we simplify the visualization by using only the first two features of the Iris dataset and classify whether a sample belongs to class 2 (`Iris-virginica`) or not, turning the problem into a binary classification.

---

## Code

```python
# Lab 2 - Logistic Regression

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
````

---

## Learning Outcome

* Understood the mathematical foundation of logistic regression and its use in binary classification.
* Gained hands-on experience in using `scikit-learn` to implement logistic regression.
* Visualized how logistic regression separates two classes using a decision boundary.
* Learned how to manipulate and split datasets for training and testing.
* Practiced plotting decision boundaries and interpreting classification outputs.

---

