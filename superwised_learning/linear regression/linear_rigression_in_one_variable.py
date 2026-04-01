import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random x values (features)
x = 2 * np.random.rand(num_samples, 1)

# Generate corresponding y values with a linear relationship (y = 4 + 3x + noise)
true_slope = 3
true_intercept = 4
noise = np.random.randn(num_samples, 1)

y = true_intercept + true_slope * x + noise

# Plotting the data
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="blue", label="Training data", alpha=0.7)
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:",x.shape)
print("Target shape is:",y.shape)

# # linear regression for y prediction 
w = 1
b = 0

def lin_reg(x, w, b):
    y_prediction = np.array([])
    for i in range(len(x)):
        y_pred = w * x[i] + b
        y_prediction = np.append(y_prediction, y_pred)
    
    return y_prediction.reshape(-1,1)

# # copute cost formula f(x) = wx+b
def compute_cost(x, y, y_prediction):
    m = len(x)
    cost = 0
    for i in range(len(y)):
        y_yi = (y_prediction[i] - y[i])
        cost = cost + (y_yi)**2
    final_cost = (1/(2*m))*cost
    return final_cost

# # compute gradient decent 
def gradient_decent(y_prediction, y, x):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    for i in range(len(x)):
        dj_dw += (y_prediction[i] - y[i]) * x[i]
        dj_db += (y_prediction[i] - y[i])
    return dj_db / m, dj_dw / m

alpha = 0.01
costs = []
for i in range(1000):
    y_prediction = lin_reg(x, w, b)
    final_cost = compute_cost(x, y, y_prediction)
    costs.append(final_cost)
    dj_db, dj_dw = gradient_decent(y_prediction, y, x)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    print(final_cost)

print(w, b)

plt.plot(costs)
plt.show()
plt.scatter(x, y, color="blue", label="Training data", alpha=0.7)
plt.plot(x, y_prediction)
plt.show()