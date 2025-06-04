import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import kdeplot

df = sns.load_dataset("titanic")
# print(df.head())
sns.set_theme()

##Imputaion Missing Values

#1- Mean Value Imputation (Better if data is distributed normally)
df['Age_mean'] = df['age'].fillna(df['age'].mean())

#2- Median Value Imputation (If we have outliers in the data)
df['Age_median'] = df['age'].fillna(df['age'].median())

fig, (ax1, ax2, ax3) =plt.subplots(1, 3)
sns.histplot(df['age'], kde = True, ax = ax1)
sns.histplot(df['Age_mean'], kde = True, ax = ax2)
sns.histplot(df['Age_median'], kde = True, ax = ax3)
ax1.set_title('Age Distribution')
ax2.set_title('Mean Age Distribution')
ax3.set_title('Median Age Distribution')
plt.tight_layout()
plt.show()


#3- Mode Imputation Technique (For categorical values)
# print(type(df['embarked']))
# print(type(df[['embarked']]))
df['Embarked_mod'] = df['embarked'].fillna(df['embarked'].mode()[0])
print(df['embarked'].isnull().sum())
print(df['Embarked_mod'].isnull().sum())



