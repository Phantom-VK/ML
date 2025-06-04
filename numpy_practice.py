import numpy as np

arr1 = np.array(((1, 2, 3, 4, 5, 5), (1, 2, 3, 3, 4, 5 )))
arr1 = arr1.reshape(6, 2)
print(arr1)
print(arr1.shape)

arr2 = np.arange(0,10)
print(arr2)
arr3 = np.ones((1, 10))
print(arr3)

# Vectorized operations
