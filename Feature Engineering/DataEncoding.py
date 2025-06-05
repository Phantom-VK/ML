import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# | Encoding Type      | Converts Categories Into      | Use When                                                                                    |
# | ------------------ | ----------------------------- | ------------------------------------------------------------------------------------------- |
# | **LabelEncoder**   | Integer values (e.g., 0,1,2)  | When **order doesn't matter**, but **algorithms do not support strings** (e.g., Tree-based) |
# | **OneHotEncoder**  | Binary columns (0/1)          | When **no order** & you want to avoid implying one (use with linear models, DL)             |
# | **OrdinalEncoder** | Integer values **with order** | When **order of categories** is meaningful (e.g., Low < Medium < High)                      |


# Sample dataset
data = pd.DataFrame({
    'City': ['Delhi', 'Mumbai', 'Chennai', 'Delhi', 'Mumbai'],
    'Department': ['HR', 'Engineering', 'Sales', 'HR', 'Sales'],
    'Experience_Level': ['Beginner', 'Intermediate', 'Expert', 'Beginner', 'Expert']
})

print("\nOriginal Data:")
print(data)

# 1. Label Encoding
"""✅ Pros:

    Simple and fast

    Good for tree-based models (they don’t assume any ordering)

❌ Cons:

 Linear models may assume numeric ordering, which could introduce bias
"""

le = LabelEncoder()
data['City_Label'] = le.fit_transform(data['City'])

print("\nLabel Encoding (City):")
print(data[['City', 'City_Label']])

# 2. One-Hot Encoding
"""
✅ Pros:

    No ordering implied

    Great for linear regression, neural networks

❌ Cons:

    Increases dimensionality

    Can lead to curse of dimensionality in large datasets
"""
ohe_data = pd.get_dummies(data['Department'], prefix='Dept')

print("\nOne-Hot Encoding (Department):")
print(ohe_data)

# 3. Ordinal Encoding (with defined order)
"""
✅ Pros:

    Preserves natural order

    Useful for models that benefit from ordinal info (e.g., Decision Trees)

❌ Cons:

    Not suitable if order doesn’t exist

    May mislead some algorithms (e.g., linear regression might interpret "Expert > Intermediate" linearly)
"""


level_order = [['Beginner', 'Intermediate', 'Expert']]
oe = OrdinalEncoder(categories=level_order)
data['Experience_Level_Ordinal'] = oe.fit_transform(data[['Experience_Level']])

print("\nOrdinal Encoding (Experience_Level):")
print(data[['Experience_Level', 'Experience_Level_Ordinal']])

#4. Target Guided Ordinal Encoding
"""
Target Guided Ordinal Encoding assigns ordinal ranks to categories based on their relationship with the target variable 
(e.g., mean target value per category). It’s ideal for:

    High-cardinality categorical features (e.g., ZIP codes, product IDs).

    Non-linear models where label encoding might mislead (unlike One-Hot, it avoids dimensionality explosion).
"""
tgo_data = pd.DataFrame({
    'City': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'Price': [100, 150, 120, 200, 180, 220, 110, 160]
})

# We calculate mean price per city
mean = tgo_data.groupby('City')['Price'].mean().to_dict()
tgo_data['encoded_price'] = tgo_data['City'].map(mean)
print("\n Target Guided Ordinal Encoding")
print(tgo_data)



