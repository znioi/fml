# FML Lab – Experiment 11: Evaluation of ML Algorithms with Balanced and Unbalanced Datasets

## 🎯 Aim

To study and implement evaluation of machine learning algorithms on **balanced** and **unbalanced** datasets, and understand how data imbalance affects model performance.

---

## 📚 Theory

### Balanced vs Unbalanced Datasets

- **Balanced Dataset:** The number of samples in each class is approximately equal.
- **Unbalanced Dataset:** One or more classes have significantly more samples than others, leading to class imbalance.

### Why It Matters

Class imbalance can cause ML models to be biased towards the majority class, resulting in misleading accuracy and poor performance on minority classes.

---

### Evaluation Metrics for Imbalanced Data

- **Accuracy:** Overall correctness but misleading on imbalanced data.
- **Precision:** Correct positive predictions / total positive predictions.
- **Recall (Sensitivity):** Correct positive predictions / total actual positives.
- **F1 Score:** Harmonic mean of precision and recall; better for imbalanced data.
- **Confusion Matrix:** Provides detailed breakdown of true/false positives and negatives.

---

## 🧪 Code Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create balanced dataset
X_bal, y_bal = make_classification(n_samples=1000, n_features=5, n_classes=2, 
                                   weights=[0.5, 0.5], random_state=42)

# 2. Create unbalanced dataset (90% class 0, 10% class 1)
X_imbal, y_imbal = make_classification(n_samples=1000, n_features=5, n_classes=2, 
                                       weights=[0.9, 0.1], random_state=42)

def train_evaluate(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n--- {title} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Evaluate on balanced dataset
train_evaluate(X_bal, y_bal, "Balanced Dataset")

# Evaluate on unbalanced dataset
train_evaluate(X_imbal, y_imbal, "Unbalanced Dataset")
