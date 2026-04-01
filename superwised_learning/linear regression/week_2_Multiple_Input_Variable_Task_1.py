import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of samples and features
num_samples = 100
num_features = 3

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, num_features)

# Define true weights (slopes) and intercept
true_weights = np.array([[3], [2], [1]])  # shape: (3, 1)
true_intercept = 4

# Generate noise
noise = np.random.randn(num_samples, 1)

# Compute y values
y_train = true_intercept + x_train @ true_weights + noise  # @ is matrix multiplication

# Plotting using only first feature for visualization
plt.figure(figsize = (8, 5))
plt.scatter(x_train[:, 0], y_train, color = "green", label = "Training data (vs 1st feature)", alpha = 0.7)
plt.title("Dummy Linear Regression Data (3 features)")
plt.xlabel("x_train[:, 0] (1st feature)")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)

# multiple linear regression 
m = len(y_train)
w = np.array([[1], [2], [3]])
b = 0.0
alpha = 0.01
# foe calculate prediction 
def prediction(w ,x_train,b):
    y_prediction = (x_train @ w + b)
    return y_prediction
# cost function
def cost_function(y_prediction,y_train):
    yp_yi = y_prediction - y_train
    final_cost = (1/(2*m)) * np.sum(yp_yi**2)
    return final_cost, yp_yi

# gradient descent
def gradient_decent(yp_yi,x_train):
    d_w = x_train.T @ yp_yi
    d_w = (d_w)/m
    d_b = np.sum(yp_yi)/m
    return d_w,d_b
costs = []
for i in range(1000):
    y_prediction = prediction(w, x_train, b)
    final_cost, yp_yi = cost_function(y_prediction, y_train)
    costs.append(final_cost)
    d_w, d_b = gradient_decent(yp_yi,x_train)
    w = w - alpha * d_w
    b = b - alpha * d_b
    print(final_cost)
plt.plot(costs)
plt.show()
print(w,b)


