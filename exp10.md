# FML Lab – Experiment 10: Classification based on Association Rules

## 🎯 Aim

To study and implement classification based on **Association Rules**, using the Apriori algorithm to find frequent itemsets and generate rules for predicting class labels.

---

## 📚 Theory

### What are Association Rules?

Association rules are if-then statements that help uncover relationships between variables in large datasets. These rules are widely used in market basket analysis but can also be applied to classification tasks.

An association rule has two key measures:

- **Support:** The proportion of records in the dataset where the itemset appears.

- **Confidence:** The likelihood that the consequent occurs given the antecedent.

Mathematically:

\[
\text{Support}(A \rightarrow B) = \frac{\text{Count}(A \cup B)}{N}
\]

\[
\text{Confidence}(A \rightarrow B) = \frac{\text{Count}(A \cup B)}{\text{Count}(A)}
\]

Where \( A \) is the antecedent and \( B \) is the consequent.

---

### Classification with Association Rules

- **Step 1:** Binarize continuous features (e.g., split into 'low' and 'high' based on quantiles).

- **Step 2:** Apply Apriori algorithm to find frequent itemsets that satisfy minimum support.

- **Step 3:** Generate association rules from these itemsets that meet minimum confidence threshold.

- **Step 4:** Use these rules to classify new instances by matching antecedents to feature values.

---

## 🧪 Code

```python
# Experiment 10 - Classification based on Association Rules

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 2. Binarize numerical features into 'low' and 'high' based on quantiles
binary_df = pd.DataFrame()
for col in df.columns[:-1]:  # Exclude the target column
    binary_df[col + '_low'] = df[col] <= df[col].quantile(1 / 3)
    binary_df[col + '_high'] = df[col] > df[col].quantile(2 / 3)

binary_df['target'] = df['target']

# 3. Apriori algorithm to find frequent itemsets based on minimum support
def apriori(data, min_sup=0.1):
    itemsets = {
        frozenset([col]): data[col].mean()
        for col in data.columns if data[col].mean() >= min_sup
    }
    return itemsets

# 4. Generate association rules from frequent itemsets based on minimum confidence
def generate_rules(frequent, min_conf=0.7):
    rules = []
    for k, s in frequent.items():
        if len(k) > 1:
            for b in k:
                antecedent = k - {b}
                consequent = b
                confidence = s / frequent.get(antecedent, 1)
                rules.append((antecedent, consequent, s, confidence))
    return rules

# 5. Find frequent itemsets and association rules
min_support = 0.1
freq_items = apriori(binary_df.iloc[:, :-1], min_sup=min_support)

min_confidence = 0.7
rules = generate_rules(freq_items, min_conf=min_confidence)

# 6. Print Frequent Itemsets
print("\nFrequent Itemsets:")
for itemset, support in freq_items.items():
    print(f"  {set(itemset)} : {support:.4f}")

# 7. Print Association Rules
print("\nAssociation Rules:")
if rules:
    for antecedent, consequent, support, confidence in rules:
        print(f"  {set(antecedent)} → {consequent} | Support: {support:.4f}, Confidence: {confidence:.4f}")
else:
    print("  No rules found")
