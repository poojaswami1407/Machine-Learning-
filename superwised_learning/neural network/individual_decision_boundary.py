import numpy as np
from matplotlib import pyplot as plt


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


X, y = load_coffee_data()
print(X.shape, y.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()


print("Features shape:", X.shape)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = np.reshape(y, (-1, 1))
print("Target shape:", y.shape)

# making neural network 

# activation functions

def relu(z):
    a = np.maximum(0,z)
    return a

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

#function for derivative of activation function 
def derivative_relu(z):     #relu 
    return (z > 0).astype(float)

def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)
    

#creating layers 
layers = [3,1]
activation = [relu, sigmoid]
derivative_activation = [derivative_relu, derivative_sigmoid]
#creating weights and bias
w = []
b = []

x_1 = X.shape[1]

for i in range(len(layers)):       # for weights
    if i == 0:
        a = np.random.rand(x_1, layers[i]) * np.sqrt(2/(x_1+layers[i]))
    else:
        a = np.random.rand(layers[i-1],layers[i]) * np.sqrt(2/(layers[i-1]+layers[i]))
    w.append(a)

for i in range(len(layers)):         # for bias
    c = np.zeros((1,layers[i]))
    b.append(c)


def forward_propogation(X, w, b):
    out = []
    out.append(X)
    z = []
    for i in range(len(layers)):
        k = ( out[i] @ w[i] ) + b[i]
        z.append(k)
        a_1 = activation[i](k)
        out.append(a_1)
    y_p = out[-1]
    return out,z,y_p


#calculating loss
m = X.shape[0]

def compute_cost(m,y_p,y):
    cost = -(y * np.log(y_p + 1e-8) + (1 - y) * np.log((1 - y_p + 1e-8)) )
    cost = np.mean(cost)
    return cost

def gradient_descent(y_p,out,y,w,b,m,z):
    d_dw = []
    d_db = []
    for i in range(len(layers)):
        if i == 0:
            d_z = (y_p - y ) 
            d_w = out[-2].T @ d_z
            d_w = d_w / m
            d_dw.insert(0,d_w)
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_b = d_b / m
            d_db.insert(0,d_b)
        else:
            d_z = d_z @ w[-i].T * derivative_activation[-(i+1)](z[-(i+1)])
            d_w = out[-(i+2)].T @ d_z
            d_dw.insert(0,d_w)
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_b = d_b / m
            d_db.insert(0,d_b)
    return d_dw,d_db

alpha = 0.5
costs = []
for i in range(10000):
    out,z,y_p = forward_propogation(X, w,b)
    cost = compute_cost(m,y_p,y)
    costs.append(cost)
    d_dw,d_db = gradient_descent(y_p,out,y,w,b,m,z)
    for j in range(len(w)):
        w[j] = w[j] - alpha * d_dw[j]
        b[j] = b[j] - alpha * d_db[j]

plt.plot(costs)
plt.show()

    
# DECISION BOUNDARY
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Flatten grid
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid
out_grid, z_grid, _ = forward_propogation(grid, w, b)

# Reshape predictions
Z_1 = z_grid[0]

z1 = Z_1[:, 0].reshape(xx.shape)
z2 = Z_1[:, 1].reshape(xx.shape)
z3 = Z_1[:, 2].reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))

plt.contour(xx, yy, z1, levels=[0], colors='r', linewidths=2)
plt.contour(xx, yy, z2, levels=[0], colors='g', linewidths=2)
plt.contour(xx, yy, z3, levels=[0], colors='b', linewidths=2)

# Plot original data
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1],
            color="red", label="Class 0")
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1],
            color="blue", label="Class 1")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary (Neural Network)")
plt.legend()
plt.show()


