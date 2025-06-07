import pandas as pd

#Download dataset from: https://github.com/krishnaik06/playstore-Dataset/blob/11c1c3fb2af5de2acb6358a48a85377b363f0b96/googleplaystore.csv
df_original = pd.read_csv("googleplaystore.csv")

#Make a copy
df = df_original.copy()

# Step 1: Understand the Data
print(df.info())
print(df.describe())

# Step 2: Handle Missing Values
print(df.shape)
print(df.isnull().sum())
#Rating has the highest number of null values, but some apps can have no ratings
#Can use following to fil them
# df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
# print(df.isnull().sum())

#Step 3: Handle Duplicates
print("__________________________________________________________________________")
print(df.duplicated())
print(df.shape)
df.drop_duplicates()

# Step 4: Fix Data Types
#Original columns which has numeric values in them
# Step 1: Select object-type columns
object_cols = df.select_dtypes(include='object').columns

# Step 2: Check if all values in the column can be converted to numeric
numeric_objects = []

for col in object_cols:
    try:
        pd.to_numeric(df[col])
        numeric_objects.append(col)
    except:
        continue

print("Object columns that are actually numeric:", numeric_objects)
