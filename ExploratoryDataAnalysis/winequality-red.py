import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with proper delimiter
df = pd.read_csv('winequality-red.csv', delimiter=';')

## DATA QUALITY CHECKS
# Check for missing values (should be 0 for this dataset)
print("Missing values per column:")
print(df.isnull().sum())

# Handle duplicates (common in wine datasets)
print(f"\nInitial duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Remaining duplicates after cleaning: {df.duplicated().sum()}\n")

## EXPLORATORY DATA ANALYSIS
# Basic statistics for numerical features
print("Descriptive statistics:")
print(df.describe())

# Correlation analysis (key for understanding relationships)
corr_matrix = df.corr()
print("\nCorrelation matrix (target='quality'):")
print(corr_matrix['quality'].sort_values(ascending=False))

## VISUALIZATION STRATEGY
# 1. Correlation Heatmap - Shows all pairwise relationships
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            annot_kws={'size':8})
plt.title('Feature Correlation Matrix', pad=20)
plt.tight_layout()
plt.show()

# 2. Distribution Plots - Understand feature distributions
plt.figure(figsize=(15,10))
for i, col in enumerate(df.columns[:-1]):  # Skip target
    plt.subplot(3,4,i+1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 3. Target Analysis - How quality relates to other features
plt.figure(figsize=(10,6))
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title('Alcohol Content by Wine Quality')
plt.show()

# 4. Pairwise Relationships (sample of key features)
sns.pairplot(df[['alcohol', 'volatile acidity', 'citric acid', 'quality']],
             hue='quality',
             plot_kws={'alpha':0.5})
plt.suptitle('Key Feature Relationships', y=1.02)
plt.show()

# 5. Multivariate Analysis - Interaction of top 3 correlated features
sns.lmplot(x='alcohol', y='sulphates',
           hue='quality',
           data=df,
           scatter_kws={'alpha':0.3},
           height=6,
           aspect=1.2)
plt.title('Alcohol vs Sulphates by Quality')
plt.show()