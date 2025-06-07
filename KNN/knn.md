# %% [markdown]
# # Diabetes Prediction using KNN with Cross-Validation and Hyperparameter Tuning

# %% [markdown]
# ## 1. Import Required Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## 2. Load and Explore the Dataset

# %%
# Load the dataset
# Note: You'll need to adjust the path to your dataset file
df = pd.read_csv('dataset.csv')

# Display first few rows
print("Dataset shape:", df.shape)
df.head()

# %%
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# %%
# Basic statistics
df.describe()

# %%
# Check the distribution of the target variable
sns.countplot(x='Outcome', data=df)
plt.title('Distribution of Diabetes Outcome')
plt.show()

# %% [markdown]
# ## 3. Data Preprocessing

# %%
# Separate features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# %%
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# %% [markdown]
# ## 4. Build KNN Model with Pipeline (Scaling + Classifier)

# %%
# Create a pipeline with StandardScaler and KNN classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# %% [markdown]
# ## 5. K-Fold Cross Validation

# %%
# Initialize K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = []
for train_idx, val_idx in kfold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Fit the model
    pipeline.fit(X_train_fold, y_train_fold)
    
    # Evaluate on validation fold
    y_pred = pipeline.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_pred)
    cv_scores.append(score)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# %% [markdown]
# ## 6. Hyperparameter Tuning with GridSearchCV

# %%
# Define parameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': range(1, 30, 2),
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1: Manhattan distance, 2: Euclidean distance
}

# %%
# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# %%
# Get the best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# %%
# Visualize the performance for different k values
results = pd.DataFrame(grid_search.cv_results_)
k_results = results[results['param_knn__weights'] == 'uniform'][results['param_knn__p'] == 2]

plt.figure(figsize=(10, 6))
plt.plot(k_results['param_knn__n_neighbors'], k_results['mean_test_score'], marker='o')
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean CV Accuracy')
plt.grid()
plt.show()

# %% [markdown]
# ## 7. Evaluate the Best Model on Test Set

# %%
# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# %%
# Evaluation metrics
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [markdown]
# ## 8. Feature Importance Analysis (Permutation Importance)

# %%
from sklearn.inspection import permutation_importance

# Compute permutation importance
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)

# %%
# Sort features by importance
sorted_idx = result.importances_mean.argsort()

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importance (test set)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Final Model Deployment

# %%
# Train final model on entire dataset with best parameters
final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=grid_search.best_params_['knn__n_neighbors'],
        weights=grid_search.best_params_['knn__weights'],
        p=grid_search.best_params_['knn__p']
    ))
])

final_model.fit(X, y)

# %%
# Example prediction
sample_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Replace with actual values
prediction = final_model.predict(sample_data)
print("Prediction for sample data:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
