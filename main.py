import numpy as np

# Numpy provides efficient storage
# It also provides better way of handling data
# It uses less memory to store data
# arr = numpy.array([1, 2, 3, 4, 5])

arr = np.array([1, 2, 3, 4, 5])
arr1 = np.array(1)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int64)
arr4 = np.array(1, ndmin=5)
print(arr)
print(arr2)
print(arr3)
print(arr4)
print(type(arr))
print("Dimensional - ", arr.ndim)
print("Dimensional - ", arr1.ndim)
print("Numpy version - ", np.__version__)
print("Numpy is used to work with arrays")
print("A dimension in arrays is one level of array depth (nested arrays).")

print(arr.dtype)
print(arr3.dtype)

# Accessing array elements -
# Accessing an array element by referring to its index number
print("Index - ")
print(arr[0])
print(arr2[0])
print(arr3[1])
print(arr3[1] + arr3[2])
print("3 element of 3 row", arr3[2, 2])
print("3D Array - ")
arr5 = np.array([[["A", "B", "C"], ["D", "E", "F"]], [["G", "H", "I"], ["J", "K", "L"]]])
print(arr5[1, 1, 2])
# Negative Indexing
# Access an array from the end by -1
print(arr5[1, -1])

# Numpy array slicing
# Slicing in python means taking elements from one given index to another given index.
# Syntax - arr_name[start:end]
# Another Syntax - arr_name[start:end:step]  # Default step - 1

arr6 = [1, 2, 3, 4, 5, 6, 7]
print(arr6[2:4])
# The result includes the start index, but excludes the end index.
print(arr6[:4])
print(arr6[4:])
# Negative slicing
print("Negative slicing -", arr6[-2:-1])
# Step slicing
print(arr6[1:7:2])
print(arr6[::2])

# Numpy Data Types
# i - integer
# b - boolean
# u - unsigned integer
# f - float
# c - complex float
# m - timedelta
# M - datetime
# O - object
# S - string
# U - unicode string
# V - fixed chunk of memory for other type ( void )

arr7 = np.array([1, 2, 3, 4, 5], dtype='S')
print(arr7.dtype)
# arr8 = np.array(["A", "1"], dtype="i")  # This will give value error
# print(arr8.dtype)
arr8 = np.array([0.2, 1.4, 2, 0])
# arr9 = arr8.astype(int)
arr9 = arr8.astype(bool)
print(arr9)

# Copy -
# It is a new array.The copy owns the data and any changes made to the copy will not affect
# original array, and any changes made to the original array will not affect the copy.
arr10 = np.array([1, 2, 3])
arr11 = arr10.copy()
arr10[2] = 4
print("Original Array -", arr10)
print("Copy Array -", arr11)
print("---------------------------------")
# View
# It is just a view of the original array. The view does not own the data and any changes made to the
# view will affect the original array, and any changes made to the original array will affect the view.
arr12 = np.array([1, 2, 3])
arr13 = arr12.view()
arr12[2] = 4
print("Original Array -", arr12)
print("Copy Array -", arr13)

# Every NumPy array has the attribute base that returns None if the array owns the data.
print(arr12.base)
print(arr13.base)

# The shape of an array is the number of elements in each dimension.
print("Shape of an array - \n", arr12.shape)
print(arr5.shape)
print(arr3.shape)

# Reshaping means changing the shape of an array.
arr14 = np.array(["A", "B", "C", "D", "E", "F", "G", "H"])
print("1D Array - ", arr14)
print("2D Array -\n", arr14.reshape(4, 2))
# The first value in reshape will determine how many arrays will create
# The first value in reshape will determine the size of each array
print("3D Array -\n", arr14.reshape(2, 2, 2))
print(arr14.base)  # Array owns the data
print("Unknown dimension -\n", arr14.reshape(4, -1))

# Flattening the array-
# It means converting a multidimensional array into a 1D array.
print("Converting array dimensional type by reshape - ")
print("Before -\n", arr3)
print("After -", arr3.reshape(-1))

# Iterating Arrays -
print("Iterating through 1-D array :")
for x in arr:
    print(x)

print("Iterating through 2-D array :")
for x in arr3:
    for y in x:
        print(y)

print("Iterating through 3-D array :")
for x in arr5:
    for y in x:
        for z in y:
            print(z)

# nditer()
print("Iteration by nditer() - ")
for x in np.nditer(arr5):
    print(x, end=" ")

# ndenumerate()
# It gives corresponding index of the element while iterating
print("\nIteration while showing index - ")
for ind, x in np.ndenumerate(arr5):
    print(ind, x)

# Joining Numpy Arrays -
# Joining means putting contents of two or more arrays in a single array.
# Join array by axes
# Syntax - np.concatenate((arr1, arr2))

arr15 = [1, 2, 3]
arr16 = [4, 5, 6]
print("Array concatenation - ")
print("Joining 1D array - ")
print(np.concatenate((arr15, arr16)))
print("Joining 2D array - ")
arr17 = np.array([[1, 2, 3], [4, 5, 6]])
arr18 = np.array([[7, 8, 9], [10, 11, 12]])
print("1 Array -\n", arr17)
print("2 Array -\n", arr18)
print("Concatenate array -\n", np.concatenate((arr17, arr18), axis=1))
print(" Joining array by Stack -")
print(np.stack((arr17, arr18), axis=1))
print("Stack along rows - ")
print(np.hstack((arr17, arr18)))  # Just like concatenate
print("Stack along columns - ")
print(np.vstack((arr17, arr18)))
print("Stack along depth(height)- ")
print(np.dstack((arr17, arr18)))

# Splitting
# It is a reverse operation of string.
# By array_split(array_name, number_of_splits)
arr19 = np.array([1, 2, 3, 4, 5, 6])
print("Splitting array -")
arr20 = np.array_split(arr19, 3)
print(arr20)
print(arr20[0])  # Accessing first element
print("Splitting 2D array -")
arr21 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(np.array_split(arr21, 2))  # How many arrays created given as argument
print("Split along rows - ")
print(np.hsplit(arr21, 2))
print("Split along columns - ")
print(np.vsplit(arr21, 2))
# print("Split along depth(height)- ")
# print(np.dsplit(arr21, 3)) -- Value Error
# dsplit only works on arrays of 3 or more dimensions

# Searching arrays -
print("Search an array by where() method ")
arr22 = np.array([1, 2, 3, 4, 5,])
print(np.where(arr22 == 4))

# Search Sorted
# It  performs a binary search in the array, and returns the index
# where the specified value would be inserted to maintain the search order.
print("Search Sorted - ")
arr23 = np.array([10, 12, 14, 16, 18, 20])
print("Index - ", np.searchsorted(arr23, 17))
# To start search from right, which gives right most index
print("Index from right - ", np.searchsorted(arr23, 19, side='right'))

# Sorting arrays -
arr24 = np.array([34, 56, 12, 28, 40])
arr25 = np.array(['B', 'A', 'D', 'C'])
print("Sorting -")
print("Sorting by number -", np.sort(arr24))
print("Sorting by alphabets - ", np.sort(arr25))

# Filter Arrays -
# Getting some elements out of an existing array and creating a new array out of them is called filtering.
# In NumPy, an array is filtered using a boolean index list.
# If the value at an index is True that element is contained in the filtered array.
# if the value at that index is False that element is excluded from the filtered array.
arr26 = np.array(['A', 1, 'B', 2])
x = [True, False, True, False]
print("Filter Arrays -")
print(arr26[x])

arr27 = np.array([10, 15, 30, 45, 60])
# 1 Method -
# arr28 = []
# for i in arr27:
#     if i % 2 == 0:
#         arr28.append(True)
#     else:
#         arr28.append(False)
# print("Even Array -", arr27[arr28])

# 2 Method -
arr28 = arr27 % 2 == 0
arr29 = arr27 % 2 != 0
print("Original Array -", arr27)
print("Even Array -", arr27[arr28])
print("Odd Array -", arr27[arr29])

