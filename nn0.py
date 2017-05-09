import numpy as np
import mathfunc

def identity_function(x):
    return x

# INPUT
X = np.array([[2.0, 3.0, 4.0], [5.0, 4.0, 3.0]])

# LayER1
W1 = np.array([[1.0, -1.0], [0.5, -0.25], [-2.0, 0.5]])
b1 = np.array([3.0, -0.5])
a1 = np.dot(X, W1) + b1
z1 = mathfunc.sigmoid(a1)

# LayER2
W2 = np.array([[0.96, -0.24], [1.25, -1.00]])
b2 = np.array([0.25, -1.25])
a2 = np.dot(z1, W2) + b2
z2 = mathfunc.sigmoid(a2)

# OUTPUT
y = identity_function(z2)

print(y)
