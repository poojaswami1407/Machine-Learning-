import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set seed
np.random.seed(42)

# Generate a classification dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,          # Only 2 useful features for visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,         # Low separation makes it harder
    flip_y=0.1,            # Add noise (10% label flipping)
    random_state=42
)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
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
layers = [16,16,1]
activation = [relu, relu, sigmoid]
derivative_activation = [derivative_relu, derivative_relu, derivative_sigmoid]
#creating weights and bias
w = []
b = []

x_1 = X.shape[1]

for i in range(len(layers)):       # for weights
    if i == 0:
        a = np.random.rand(x_1, layers[i]) * 0.05
    else:
        a = np.random.rand(layers[i-1],layers[i]) * 0.05
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

alpha = 0.01
costs = []
for i in range(1000):
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
_, _, y_grid_pred = forward_propogation(grid, w, b)

# Reshape predictions
Z = y_grid_pred.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z > 0.5, alpha=0.3)

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