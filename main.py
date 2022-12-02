import numpy as np
from numpy import random

"""
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


# ----------------------Random----------------------------
# Random means something that can not be predicted logically.
# Random numbers generated through a generation algorithm are called pseudo random.

# Example -
print(random.randint(990, 1000))  # randint(low, high)
# This will generate an integer number

print(random.rand())
# This will generate a float number between 0 and 1

# Generating a random array -
print("Random array - ", random.randint(100, size=(10)))
# The randint() method takes a size parameter where you can specify the shape of an array.

print("Generating a 2D array -")
print("2D Array -\n", random.randint(100, size=(3, 3)))

print("Generating a 2D float array -")
print("2D Float Array -\n", random.rand(3, 5))

# The choice() method also allows you to return an array of values.
# Syntax - random.choice([-, -, -]. size=(-, -))

print("Random array by choice method - ")
print("Choice -\n", random.choice([5, 2, 4, 1, 3], size=(5, 5)))

print("Data Distribution -")
print("It is a list of all possible values, and how often each value occurs.")
print("Syntax- random.choice([-,-,-], p=[-,-,-], size=(-)")
# The sum of all probability numbers should be 1

print("Data -", random.choice([10, 20, 30, 40, 50], p=[0.2, 0.3, 0.1, 0.4, 0.0], size=20))
print("In the above data 50 will never occur as its probability is given 0")
print("2D Array -")
print("Data -\n", random.choice([10, 20, 30, 40, 50], p=[0.2, 0.3, 0.1, 0.4, 0.0], size=(4, 5)))

print("Random permutations - ")
print("A permutation refers to an arrangement of elements.")
print("The NumPy Random module provides two methods for this: shuffle() and permutation().")
arr30 = np.array([23, 25, 27, 29])
arr31 = np.array([24, 26, 28, 30])

print("1. Shuffle means changing arrangement of elements in-place. i.e. in the array itself.")
print("Array -", arr30)
random.shuffle(arr30)
print("Shuffled Array -", arr30)
print("The shuffle() method makes changes to the original array.")

print("2. Permutation")
random.permutation(arr31)
print("After calling permutation array -", arr31)
print("The permutation() method returns a re-arranged array (and leaves the original array un-changed).")

print("Sum of array elements -", arr31.sum())
print("Zeroes Array -\n", np.zeros((3, 3)))
temp = arr31[arr31 % 2 == 0]
print("Conditional Formatting -", temp)

print("Operations on array -")
print("Original array -", arr31)
print("Addition by 1 -", arr31+1)
print("Subtraction by 1 -", arr31-1)
print("Division by 2 -", arr31//2)
print("Multiplication by 2 -", arr31*2)
print("Squaring each element by 2 -", arr31**2)
arr32 = np.array([[1, 2, 3], [0, 9, 8], [4, 5, 6]])
print("Largest element is -", arr32.max())
print("Binary operators")
arr33 = np.array([[0, 9, 8], [4, 5, 6], [1, 2, 3]])
print("Array sum -\n", arr32+arr33)
print("Array multiplication -\n", arr32*arr33)

# Iteration Over Array -
# nditer - It is an efficient multidimensional iterator object using which it is possible to iterate over an array.
print("Iteration")
arr34 = np.arange(12)
print(arr34.reshape(2, 6))
for x in np.nditer(arr34):
    print(x, end=" ")
print("\nFortran order -")
for x in np.nditer(arr34, order='F'):
    print(x, end=" ")
# for x in np.nditer(arr34, op_flags = ['readwrite']):
#     x[...] = 5*x


# Binary Operations -
# Binary operators acts on bits and performs bit by bit operation.
n1 = 10
n2 = 11
# 1. numpy.bitwise_and() - This function is used to Compute the bit-wise AND of two array element-wise.
print("Bitwise_and -", np.bitwise_and(n1, n2))
# 2. numpy.bitwise_or() - This function is used to Compute the bit-wise OR of two array element.
print("Bitwise_or -", np.bitwise_or(n1, n2))
# 3. numpy.bitwise_xor() - This function is used to Compute the bit-wise XOR of two array element.
print("Bitwise_xor -", np.bitwise_xor(n1, n2))
# 4. numpy.binary_repr(number, width=None)
print("Binary representation of ", n1, "is", np.binary_repr(n1))  # optional parameter - width=5
print("Binary representation of ", n2, "is", np.binary_repr(n2))

# ---------------String Operations--------------------------
# 1. numpy.char.lower()
print("Lowercase string -", np.char.lower('DHAIRYA'))

# 2. numpy.char.split()
print("Split string -", np.char.split("D*S", sep="*"))

# 3. numpy.char.join()
print("Join string -", np.char.join('--', 'DHAIRYA'))

# 4. numpy.char.count()
print("Count -", np.char.count('DHAIRYA', 'A'))

# 5. numpy.char.rfind()  # -1 for not found
print("String Present at -", np.char.rfind("DHAIRYA", 'I'))

# 6. numpy.char.isnumeric()
print("Numeric characters -", np.char.isnumeric("1122"))

# 7. numpy.char.equal()
print("Characters are equal -", np.char.equal("DHAIRYA", "DHAIRYA"))

# 8. numpy.char.not_equal()
print("Characters are not equal -", np.char.not_equal("DHAIRYA", "DHAIRYA"))

"""

# --------------------------Sorting---------------------------------
arr35 = np.array([[10, 20], [5, 7]])
print("Sorting along first axis -\n", np.sort(arr35, axis=0))
print("Sorting along last axis -\n", np.sort(arr35, axis=-1))
print("Sorting along none axis -\n", np.sort(arr35, axis=None))

# numpy.argsort() : This function returns the indices that would sort an array.
arr36 = np.array([13, 15, 12, 14, 11, 16, 10])
print('Sorted indices of original array->', np.argsort(arr36))

# --------------------------Searching-------------------------------

# np.max()
print("Max element -", np.max(arr36))

# np.argmax()
print("Index of max element -", np.argmax(arr36, axis=0))

# np.min()
print("Max element -", np.min(arr36))

# np.argmin()
print("Index of min element -", np.argmin(arr36, axis=0))

# np.count_nonzero
print("Number of non zero values -", np.count_nonzero([0, 1, 0, 2, 0, 3]))

# ------------------------------Random Sample-------------------------------
# Syntax : numpy.random.random_sample(size=None)
print("Random Sample values -", np.random.random_sample(2))







