import numpy as np

alpha = 1.4
beta = -0.3
gamma = 1.7
theta = -0.7
omega = -128.0
convert_matrix = np.array([
    [gamma,1,0,0],
    [beta,1,theta,0],
    [0,1,alpha,0],
    [gamma,0,0,1],
    [beta,0,theta,1],
    [0,0,alpha,1]
    ])
print(convert_matrix.shape)
#data_len = 4000000
data_len = 32
unit_size = 4
a = np.array(range(data_len))

w = np.zeros((data_len))
for i in range(0,data_len,2):
    w[i] = omega

print(w)
print(w.shape)
print(a.shape)
print(a)
#a = a - w
#print(a)
b = np.reshape(a,(len(a)//unit_size,unit_size))
print(b.shape)
print(b)


c = np.dot(convert_matrix,b.T).T
c = np.clip(c, 0, 255)
c = c.astype(np.uint8)
print(c.shape)
print(c)

d = np.reshape(c,(4,12))
print(d.shape)
print(d)

e = np.reshape(d,(8,2,3))
print(e.shape)
print(e[0,0,:])
print(e[0,1,:])
