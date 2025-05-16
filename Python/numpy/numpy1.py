import numpy as np
import random
array=np.array([[1,2,3,4,5,],[2,3,4,5,6]],dtype='int32')
print(array)
print(array.ndim)
print(array.dtype)
print(array[1][2])
print(array[0,:])   #get a specific row   : means selcting all rows or columns
print(array[:,3])   #get a specific column
print(array[0,0:4:2])
print(array[1,1:4:1])
array[1,4]=20   #here this is accessing the first row and 4 th column and this is same as array[1][5]
print(array)
array=np.zeros((3,4))   #also np.ones()
print(array)
array=np.full((3,4),44)
print(array)
array=np.random.rand(3,4)    #it makes a 2d array of 3 rows and 4 columns and it fills it with the random float
print(array)
array=np.random.randint(7,size=(3,4))
print(array)
array=np.identity(5)   #this makes an identity matrix  of the size 5
print(array)
array=np.array([[1,2,3]])
r1=np.repeat(array,3,axis=0)
print(r1)
b=array.copy()
print(b)
array=np.array([[1,2,3],[4,5,6]])
print("df")
array=array+2
print(array)
array=array*2
print(array)
#print(np.sin(array))
array=array**2
print(array)
#linear algebra
a=np.full((2,3),77)
b=np.full((3,2),88)
print(np.matmul(a,b))

array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.linalg.det(array))
print(np.linalg.inv(array))
print(np.linalg.eigvals(array))
print(np.linalg.eigvals(array))
print(array.shape)
newarray=array.reshape((9,1))
print("hii")
print(newarray)

#vertical stacking of the arrays

v1=np.array([1,2,3,4])
v2=np.array([5,6,7,8])
v3=np.vstack([v1,v2])
print(v3)
print(np.hstack([v1,v2]))
#also we can stack like np.vstack([v1,v2.v2.v2,v1])

#importing file and converting into the numpy file

newdata=np.genfromtxt(('data.txt'),delimiter=',')
print(newdata)

#boolean masking and advanced indexing
print(newdata>50)

import matplotlib.pyplot as plt
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
