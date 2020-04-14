import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gauss(x,l):    
    z = np.zeros(len(x))
    for i in range(len(x)):
        z[i] = np.exp((-1) * np.linalg.norm(x[i]-l)**2 / 2) / np.sqrt(2 * np.pi)
    return z

n = 300
r = np.random.rand(n)
t = np.random.rand(n) * 2 * np.pi

x = r * np.cos(t)
y = r * np.sin(t)

X = np.c_[x,y]

print(X)

X1 = X[r < 0.5]
X2 = X[r >= 0.5]

l1 = X[np.random.randint(0, len(X))]
l2 = X[np.random.randint(0, len(X))]
l3 = X[np.random.randint(0, len(X))]

print(l1)
print(l2)
print(l3)

z1 = gauss(X,l1)
z2 = gauss(X,l2)
z3 = gauss(X,l3)

Z = np.c_[z1,z2,z3]

print(Z)

Z1 = Z[r < 0.5]
Z2 = Z[r >= 0.5]


plt.scatter(X1[:,0], X1[:,1], s=50, c='r', marker='.')
plt.scatter(X2[:,0], X2[:,1], s=50, c='b', marker='.')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()

print(Z.shape)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot(Z1[:,0], Z1[:,1], Z1[:,2], ".", ms=10,c='r')
ax.plot(Z2[:,0], Z2[:,1], Z2[:,2], ".", ms=10,c='b')
plt.show()