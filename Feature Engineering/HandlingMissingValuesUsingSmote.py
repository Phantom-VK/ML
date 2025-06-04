"""### **4. Key Takeaways**
1. **Features (`n_features`)**  
   - Columns in the dataset (e.g., `Feature 1`, `Feature 2` in the plot).
   - Only `n_informative` features truly impact the class labels.

2. **Classes (`n_classes`)**  
   - The target variable (e.g., binary `0` or `1` in the plot).

3. **Clusters (`n_clusters_per_class`)**  
   - Subgroups within a class (e.g., red class has 2 dense regions).

---

### **5. Real-World Analogy**
Imagine classifying **animals**:
- **Features**: `weight`, `height`, `fur_length`.
- **Classes**: `0 = Cat`, `1 = Dog`.
- **Clusters per Class**:  
  - Cats: `Small Cats` (cluster 1), `Big Cats` (cluster 2).  
  - Dogs: `Small Dogs` (cluster 1), `Big Dogs` (cluster 2).

The `make_classification()` function mimics this structure synthetically.

---

### **6. When to Adjust These Parameters?**
- Increase `n_clusters_per_class` if you want more complex class distributions (e.g., multiple subtypes per class).
- Increase `n_informative` if you need more features to influence the class.
- Use `n_redundant` to simulate correlated features (e.g., `weight` and `size` might be redundant)."""


##X = Independent Feature
##y = dependent feature
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Generate synthetic imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,  # Explicitly state informative features
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.90],  # 90% class 0, 10% class 1
    random_state=12
)

# Combine into a DataFrame
final_df = pd.DataFrame(X, columns=['f1', 'f2'])
final_df['target'] = y

# Check class distribution
print("Before SMOTE:")
print(final_df['target'].value_counts())

# Apply SMOTE
oversample = SMOTE(random_state=12)  # Set random_state for reproducibility
X_after, y_after = oversample.fit_resample(final_df[['f1', 'f2']], final_df['target'])

print("\nAfter SMOTE:")
print(pd.Series(y_after).value_counts())

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
ax1.scatter(final_df['f1'], final_df['f2'], c=final_df['target'], cmap='coolwarm', alpha=0.6)
ax1.set_title("Before SMOTE (Imbalanced)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")

# After SMOTE
ax2.scatter(X_after['f1'], X_after['f2'], c=y_after, cmap='coolwarm', alpha=0.6)
ax2.set_title("After SMOTE (Balanced)")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
