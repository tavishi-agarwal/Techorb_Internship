import numpy as np

l = [1, 2, 3, 4, 0]             # Define a list
my_array = np.array(l)         # Pass the list to np.array()

type(my_array)

second_list = [5, 6, 7, 8, 9]

two_d_array = np.array([my_array, second_list])

print(two_d_array)

two_d_array.shape

two_d_array.size

two_d_array.dtype

np.identity(n = 5)

np.eye(N = 3,  # Number of rows
       M = 5,  # Number of columns
       k = 2)  # Index of diagonal for starting

np.ones(shape= [2, 4])

np.zeros(shape= [4, 6])

# Indexing and slicing of 1D array is similar to python list

l = [1, 2, 3, 4, 0]             
my_array = np.array(l)
two_d_array = np.array([my_array, my_array + 6, my_array + 12])
print(two_d_array) 

two_d_array[1, 4]

two_d_array[1:, 4:]

two_d_array[::-1, ::-1]
