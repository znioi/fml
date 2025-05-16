
### 📄 `README.md` – **Experiment 7: Decision Trees**


# FML Lab – Experiment 7: Decision Trees

## 🎯 Aim

To study and implement **Decision Tree** classifier for multi-class classification using the Wine dataset, and visualize the tree.

---

## 📚 Theory

### What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm used for classification and regression tasks. It splits the data based on feature values to create a tree-like model of decisions.

- Each **internal node** tests a feature.
- Each **branch** corresponds to an outcome of the test.
- Each **leaf node** represents a class label (classification) or a value (regression).

---

### How does a Decision Tree work?

The tree is built by recursively splitting the dataset into subsets based on the feature that results in the best split.

**Common criteria to choose the best split:**

- **Gini Impurity**: Measures the probability of incorrectly classifying a randomly chosen element.

  Gini impurity for a node with classes \( c \) is:

```

Gini = 1 - Σ (p\_i)^2

````

where \( p_i \) is the proportion of class \( i \) instances in the node.

- **Entropy and Information Gain**: Entropy measures the disorder in the dataset, and splits are chosen to maximize information gain (reduction in entropy).

---

### Advantages

- Easy to understand and interpret.
- Handles both numerical and categorical data.
- Requires little data preprocessing.

---

## 🧪 Code

```python
# Lab 7 - Decision Trees

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load Wine dataset
wine = datasets.load_wine()

# 2. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
  wine.data, wine.target, test_size=0.3, random_state=42
)

# 3. Train Decision Tree Classifier with max depth 3
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# 4. Evaluate accuracy
accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
print(f"Accuracy: {accuracy:.2f}")

# 5. Visualize the Decision Tree
plt.figure(figsize=(15,10))
plot_tree(
  dt_classifier,
  feature_names=wine.feature_names,
  class_names=wine.target_names,
  filled=True,
  rounded=True
)
plt.show()
````

---

## ✅ Learning Outcomes

* Learned the basic working principle of Decision Trees.
* Understood how splitting criteria like Gini impurity affect tree building.
* Applied Decision Trees to a real multi-class dataset.
* Visualized the trained decision tree for interpretability.
* Evaluated model accuracy on unseen test data.


