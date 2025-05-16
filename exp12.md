# FML Lab – Experiment 12: Comparison of Machine Learning Algorithms Based on Different Parameters

## 🎯 Aim

To compare the performance of different machine learning algorithms by varying key parameters and analyzing their impact on model accuracy and other evaluation metrics.

---

## 📚 Theory

### Why Compare Algorithms?

Different ML algorithms have unique strengths and weaknesses depending on the dataset and parameter settings. Comparing them helps select the best model for a specific problem.

### Key Parameters Affecting ML Models

- **For Logistic Regression:** Regularization strength (`C`), solver type.
- **For K-Nearest Neighbors (KNN):** Number of neighbors (`k`), distance metric.
- **For Decision Trees:** Maximum depth, minimum samples per leaf.
- **For Support Vector Machines (SVM):** Kernel type, regularization parameter `C`, gamma.
- **For Random Forest:** Number of trees (`n_estimators`), max depth.

### Metrics to Compare

- Accuracy
- Precision, Recall, F1-score
- Training time
- Model complexity (depth, number of parameters)

---

## 🧪 Code Example

```python
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Define models and parameters to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3),
    "SVM (linear kernel)": SVC(kernel='linear'),
    "Random Forest (n_estimators=100)": RandomForestClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    elapsed = end - start
    results[name] = {"Accuracy": accuracy, "Training Time (s)": elapsed}

# Display results
print("Model Performance Comparison:\n")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy = {metrics['Accuracy']:.3f}, Training Time = {metrics['Training Time (s)']:.4f}s")
