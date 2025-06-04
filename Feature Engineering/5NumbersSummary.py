import seaborn as sns
import matplotlib.pyplot as plt

list_marks = [23, 34, 45, 56, 76, 86, 34, 56, 89, 34, 56, 88, 78, -100]

# Plot
plt.figure(figsize=(8, 4))
sns.boxplot(x=list_marks)
plt.title("Boxplot of Marks (Outliers Visible)")
plt.show()

import numpy as np

minimum, q1, median, q3, maximum = np.quantile(list_marks, [0, 0.25, 0.50, 0.75, 1.0])
IQR = q3 - q1
lower_fence = q1 - 1.5 * IQR #Anything lower than lower fence will be considered as outlier
upper_fence = q3 + 1.5 * IQR #Anything higher than higher fence will be considered as outlier

print(f"""
Minimum: {minimum}
Q1 (25th percentile): {q1}
Median: {median}
Q3 (75th percentile): {q3}
Maximum: {maximum}
IQR: {IQR}
Lower Fence: {lower_fence}
Upper Fence: {upper_fence}
""")

# Option 1: Remove Outliers

cleaned_marks = [x for x in list_marks if x >= lower_fence and x <= upper_fence]
print("Cleaned Data:", cleaned_marks)

# Option 2: Cap Outliers (Winsorization)
capped_marks = np.clip(list_marks, a_min=lower_fence, a_max=upper_fence)
print("Capped Data:", capped_marks)

# Option 3: Replace with Median
median = np.median(list_marks)
replaced_marks = [median if x < lower_fence or x > upper_fence else x for x in list_marks]
print("Median-Replaced Data:", replaced_marks)