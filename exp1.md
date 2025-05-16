# FML Lab – Experiment 1: Linear Regression

## 🎯 Aim
To study and implement Linear Regression for predicting continuous target variables based on one or more features.

---

## 📚 Theory

Linear Regression is one of the simplest and most widely used supervised learning algorithms for regression problems. It models the relationship between a dependent variable \( y \) and one or more independent variables \( X \) by fitting a linear equation:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
\]

Where:
- \( y \) is the predicted output,
- \( \beta_0 \) is the intercept (bias),
- \( \beta_1, \beta_2, \ldots, \beta_n \) are coefficients (weights) for each feature,
- \( \epsilon \) is the error term.

The goal is to find the best-fitting line by minimizing the **Mean Squared Error (MSE)** between predicted and actual values:

\[
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
\]

Where \( m \) is the number of training samples.

---

## 🧪 Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model and train
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()
plt.show()
