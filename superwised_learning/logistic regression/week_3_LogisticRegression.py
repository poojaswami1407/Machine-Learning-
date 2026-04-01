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

#creating weights and bias
w = np.array([[0], [0]])
b = 0 

# prediction
def logistic_regression(x, w, b):
    z_i = x @ w + b
    z = 1 / (1 + np.exp(-z_i)) + 1e-8
    return z

#compute cost function 
cost = 0 
m = 200

def compute_cost(z, y, m):
    cost = -y * np.log(z + 1e-8) - (1 - y) * np.log((1 - z) + 1e-8)
    cost = np.sum(cost)
    final_cost = cost / m
    return final_cost 

# compute gradient decent 
def gradient_descent(w, b, y, z):
    d_w = x.T @ (z - y)
    d_dw = d_w / m 
    d_b = (z - y)
    d_b = np.sum(d_b)
    d_db = d_b / m 
    return d_dw, d_db

alpha = 0.01
costs = []

#iteration for updating w and b
for i in range(10000):
    z = logistic_regression(x, w, b)
    final_cost = compute_cost(z, y, m)
    costs.append(final_cost)
    d_dw, d_db = gradient_descent(w, b, y, z)
    w = w - alpha * d_dw
    b = b - alpha * d_db
    print(final_cost)
print(w,b)
plt.plot(costs)
plt.show()

w_1 = w[0]
w_2 = w[1]
print(w_1, w_2)
x_1 = x[:,0]
print(x_1.shape)
 
#decision boundary 
x_2 = (-b - (w_1 * x_1)) / w_2
print(x_2.shape)

y = y.ravel()
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.plot(x_1,x_2)
plt.show()








