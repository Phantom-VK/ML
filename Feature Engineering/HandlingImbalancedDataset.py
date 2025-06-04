import pandas as pd
import numpy as np

np.random.seed(42) # Seed is used for reproducibility
#
# print(np.random.randint(0, 15))
# # np.random.seed(42)
# print(np.random.randint(0, 15))

# We create a sample dataset with two classes
total_samples = 1000
class_1_sample_ratio = 0.9
num_class_1 = int(total_samples * class_1_sample_ratio)
num_class_0 = total_samples - num_class_1
# print(num_class_0, num_class_1)

class_1 = pd.DataFrame(
    {
        "feature1":np.random.normal(size=num_class_1),
        "feature2":np.random.normal(size=num_class_1),
        "target":[1]*num_class_1
    }
)

class_0 = pd.DataFrame(
    {
        "feature1":np.random.normal(size=num_class_0),
        "feature2":np.random.normal(size=num_class_0),
        "target":[0]*num_class_0
    }
)
df = pd.concat([class_1, class_0]).reset_index(drop=True)
print(df['target'].value_counts())

#Upsampling
from sklearn.utils import resample

df_minority = df[df['target'] == 0]
df_majority = df[df['target'] == 1]


df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority),
                                 random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled['target'].value_counts())

#Same for downsampling, just use majority dataset instead of minority