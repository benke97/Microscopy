import numpy as np
import matplotlib.pyplot as plt

V = np.array([[1,1], [-2,2], [4,-7]])
origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point
print(np.size(V,axis=0))
print(origin)
print(V)

plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'])
plt.show()

x = np.array([[1,2],[3,4]])
np.repeat(x, [1, 2], axis=0)
print(np.repeat(x, [1, 3], axis=0))

origin2 = np.repeat(np.array([[0],[0]]),np.size(V,axis=0),axis=1)
print(origin2)