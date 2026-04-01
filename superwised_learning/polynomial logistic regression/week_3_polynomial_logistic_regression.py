import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# Set seed
np.random.seed(42)

# Generate a classification dataset
x, y = make_classification(
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
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape:", x.shape)
print("Target shape:", y.shape)

y = y.reshape(-1, 1)
print("Target shape:", y.shape)


x_1 = x[:,[0]]
x_2 = x[:,[1]]
x_3 = x_1**2
x_4 = x_2**2
x_matrix  = np.hstack((x_1, x_2, x_3, x_4))
print(x_matrix.shape)

# creating weights and bias
w = np.array([[0], [0], [0], [0]])
b = 0 

# compute prediction 
def logistic_regression(x, w, b):
    z_i = x @ w + b
    z = 1 / (1 + np.exp(-z_i)) + 1e-8
    return z

#cost function 
cost = 0 
m = 200
lembda = 100

def compute_cost(z, y, m):
    cost = -y * np.log(z + 1e-8) - (1 - y) * np.log((1 - z) + 1e-8) 
    cost = np.sum(cost)
    cost = cost / m
    w_2 = np.sum(w**2)
    final_cost = cost + ((lembda / (2 * m)) * w_2)
    return final_cost 

# compute gradient decent 
def gradient_descent(w, b, y, z, x):
    d_w = x.T @ (z - y)
    d_w = d_w / m 
    w_sum = np.sum(w)
    d_dw = d_w + ((lembda / m) * w_sum )
    d_b = (z - y)
    d_b = np.sum(d_b)
    d_db = d_b / m 
    return d_dw, d_db

alpha = 0.01
costs = []

#iteration for updating w and b
for i in range(10000):
    z = logistic_regression(x_matrix, w, b)
    final_cost = compute_cost(z, y, m)
    costs.append(final_cost)
    d_dw, d_db = gradient_descent(w, b, y, z, x_matrix)
    w = w - alpha * d_dw
    b = b - alpha * d_db
    print(final_cost)
print(w,b)
plt.plot(costs)
plt.show()

# decision boundary 
# Decision Boundary using grid

# Create grid
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Prepare polynomial features for grid
grid = np.c_[xx.ravel(), yy.ravel()]
grid_poly = np.c_[grid[:, 0], grid[:, 1], grid[:, 0]**2, grid[:, 1]**2]

# Predict on grid
zz = logistic_regression(grid_poly, w, b)
zz = zz.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)

# Plot original points
plt.scatter(x[y.flatten()==0][:,0], x[y.flatten()==0][:,1], alpha=0.6, label="Class 0")
plt.scatter(x[y.flatten()==1][:,0], x[y.flatten()==1][:,1], alpha=0.6, label="Class 1")

plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()



